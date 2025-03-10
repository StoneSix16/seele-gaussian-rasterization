/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "forward.h"
#include "auxiliary.h"
namespace cg = cooperative_groups;

#define attr_off_id 0
#define attr_off_xy 4
#define attr_off_con_o 8
#define attr_off_feature 12

// Identify address of Gaussian attr each thread should load 
__forceinline__ __device__ const int* set_load_info(
	const int idx, 
	const uint32_t* __restrict__ point_list,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	int& stride, 
	bool& load_id)
{
	const int* ret;
	if(idx == 4 || idx == 5)
	{
		ret = reinterpret_cast<const int*>(points_xy_image) + (idx-4);
		stride = 2;
	}
	else if(idx >= 8 && idx < 12)
	{
		ret = reinterpret_cast<const int*>(conic_opacity) + (idx-8);
		stride = 4;
	}
	else if(idx >= 12 && idx < 15)
	{
		ret = reinterpret_cast<const int*>(features) + (idx-12);
		stride = 3;
	}
	else
	{
		ret = reinterpret_cast<const int*>(point_list);
		stride = 1;
	}
	load_id = (idx == 0);
	return ret;
}

__forceinline__ __device__ float compute_alpha(const float opacity, const float power)
{
	float alpha = min(0.99f, opacity * __expf(power));
	return alpha;
}

__forceinline__ __device__ int blend_batch(
	const int to_render, const float2 pixf, const float2 pix_cent, const int thread_rank,
	const int* data, int* precomp_alpha,
	bool& done, float& T, float3& C
	)
{
	int contrib = -1;
	float4 rgbd;

	for(int render_progress = 0; render_progress < to_render; render_progress+=GROUP_SIZE)
	{
		int index = render_progress+(thread_rank%GROUP_SIZE);
		float2 xy = *reinterpret_cast<const float2*>(data + (index*GAUSSIAN_SIZE+attr_off_xy));
		float4 con_o = *reinterpret_cast<const float4*>(data + (index*GAUSSIAN_SIZE+attr_off_con_o));

		float2 d = { xy.x - pix_cent.x, xy.y - pix_cent.y };
		float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
		float alpha = compute_alpha(con_o.w, power);
		precomp_alpha[thread_rank] = alpha < 1.0f / 255.0f;
		__syncwarp();
		
		if(!done)
		{
			for (int i = 0; i < min(to_render - render_progress, GROUP_SIZE); i++)
			{
				if (precomp_alpha[(thread_rank&(~(GROUP_SIZE-1))) + i]) continue;
				index = (render_progress+i);

				rgbd = *reinterpret_cast<const float4*>(data + (index*GAUSSIAN_SIZE+attr_off_feature));
				con_o = *reinterpret_cast<const float4*>(data + (index*GAUSSIAN_SIZE+attr_off_con_o));
				xy = *reinterpret_cast<const float2*>(data + (index*GAUSSIAN_SIZE+attr_off_xy));
				
				d = { xy.x - pixf.x, xy.y - pixf.y };
				power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
				alpha = compute_alpha(con_o.w, power);

				C.x += rgbd.x*alpha*T;
				C.y += rgbd.y*alpha*T;
				C.z += rgbd.z*alpha*T;
				contrib = index;

				T *= (1 - alpha);
				if(T < 0.0001f) done = true;
			}
		}
	}
	return contrib + 1;
} 

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_CR(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t lane = block.thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = block.thread_rank() / WARP_SIZE;
	// Each warp load 2 Gaussians
	constexpr uint32_t BATCH_SIZE = 2*BLOCK_SIZE/WARP_SIZE;
	constexpr int BATCH_OFF = 2*BATCH_SIZE;
	int stride;
	bool load_id;
	const int* gaussian_attribute = set_load_info(
		block.thread_rank()%GAUSSIAN_SIZE, 
		point_list,
		points_xy_image,
		features,
		conic_opacity,
		stride, 
		load_id);

	// the placement of threads in a warp with 2*2 window size
	// 00 01 04 05 16 17 20 21
	// 02 03 06 07 18 19 22 23
	// 08 09 12 13 24 25 28 29
	// 10 11 14 15 26 27 30 31
	constexpr uint X_size = 8 / GROUP_X;

	const uint midrank = lane / GROUP_SIZE;
	const uint midx = midrank%X_size*GROUP_X;
	const uint midy = midrank/X_size*GROUP_Y;
	const uint headrank = lane % GROUP_SIZE;
	const uint headx = headrank%GROUP_X;
	const uint heady = headrank/GROUP_X;
	const uint2 pix_min = { 
		block.group_index().x*BLOCK_X + (warp_idx%2)*8 + midx,
		block.group_index().y*BLOCK_Y + (warp_idx/2)*4 + midy};
	const uint2 pix = { 
		pix_min.x + headx, 
		pix_min.y + heady};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const float2 pix_cent = { 
		(float)(pix_min.x) + (float)(GROUP_X-1.0f)/2.0f, 
		(float)(pix_min.y) + (float)(GROUP_Y-1.0f)/2.0f };

	// Check if this thread is associated with a valid pixel or outside.
	const bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[block.group_index().y * ((W + BLOCK_X - 1) / BLOCK_X) + block.group_index().x];
	bool done = !inside || range.x >= range.y;

	// Initialize helper variables
	float T = 1.0f;
	float3 C = {0.0f, 0.0f, 0.0f};
	float expected_invdepth = 0.0f;
	uint32_t last_contributor = 0;
	int num_blended;

	// double-buffer for Gaussian attributes loading
	__shared__ int collected_data[2][BATCH_SIZE*GAUSSIAN_SIZE];
	// buffer for alpha of pixel groups
	__shared__ int precomp_alpha[BLOCK_SIZE];

	uint32_t off = 0, stage = 0;
	const int gaussian_idx = block.thread_rank()/GAUSSIAN_SIZE;
	const int gaussian_attr_idx = block.thread_rank();
	const int gaussian_id_idx = __shfl_sync(~0, gaussian_attr_idx, 0, 16);
	int progress = range.x;

#define Blend(to_render, attr_data) blend_batch(to_render, pixf, pix_cent, block.thread_rank(), attr_data, precomp_alpha, done, T, C); 

	if(range.y - range.x < BATCH_SIZE)
	{
		off = load_id ? min(progress + gaussian_idx, range.y-1) : 0;
		collected_data[0][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
		block.sync();

		off = load_id ? off : collected_data[0][gaussian_id_idx];
		collected_data[0][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
		block.sync();
		last_contributor = Blend(range.y - range.x, collected_data[0]);
	}
	else
	{
		off = load_id ? range.x + gaussian_idx : 0;
		progress += BATCH_SIZE;
		collected_data[0][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
		block.sync();

		off = load_id ? min(progress + gaussian_idx, range.y-1) : collected_data[0][gaussian_id_idx];
		collected_data[1][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
		
		progress += BATCH_SIZE;
		
		while(progress < range.y)
		{
			if (__syncthreads_and(done)) break;

			off = load_id ? min(progress + gaussian_idx, range.y-1) : collected_data[stage^1][gaussian_id_idx];
			collected_data[stage][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
			
			num_blended = Blend(BATCH_SIZE, collected_data[stage^1]);
			if(num_blended > 0) last_contributor = progress - BATCH_OFF + num_blended;
			stage ^= 1;
			progress += BATCH_SIZE;
		}

		if (!__syncthreads_and(done))
		{
			off = load_id ? off : collected_data[stage^1][gaussian_id_idx];
			collected_data[stage][gaussian_attr_idx] = __ldg(gaussian_attribute + off*stride);
			
			// range.y - 2 batch
			num_blended = Blend(BATCH_SIZE, collected_data[stage^1]);
			if(num_blended > 0) last_contributor = progress - BATCH_OFF + num_blended;
			block.sync();
			num_blended = Blend(-progress + BATCH_SIZE + range.y, collected_data[stage]);
			if(num_blended > 0) last_contributor = progress - BATCH_SIZE + num_blended;
		}
	}

#undef Blend

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		out_color[0 * H * W + pix_id] = C.x + T * bg_color[0];
		out_color[1 * H * W + pix_id] = C.y + T * bg_color[1];
		out_color[2 * H * W + pix_id] = C.z + T * bg_color[2];
		n_contrib[pix_id] = last_contributor - range.x;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_mark(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ visible_gaussians)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	uint32_t top_gaussians[MAX_GS];
	float top_gaussians_score[MAX_GS];
	int min_score_idx = 0;
	int top_gaussians_size = 0;

	top_gaussians_score[min_score_idx] = 520.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float weight = alpha * T;
			if (top_gaussians_size < MAX_GS)
			{
				top_gaussians[top_gaussians_size] = collected_id[j];
				top_gaussians_score[top_gaussians_size] = weight;
				atomicAdd(&visible_gaussians[collected_id[j]], 1);
				if (weight < top_gaussians_score[min_score_idx])
					min_score_idx = top_gaussians_size;
				top_gaussians_size++;
			} else if (weight > top_gaussians_score[min_score_idx])
			{
				// remove
				atomicSub(&visible_gaussians[top_gaussians[min_score_idx]], 1);
				// insert
				top_gaussians[min_score_idx] = collected_id[j];
				top_gaussians_score[min_score_idx] = weight;
				atomicAdd(&visible_gaussians[collected_id[j]], 1);
				// update min_score_idx
				for (int k = 0; k < MAX_GS; k++)
				{
					if (top_gaussians_score[k] < top_gaussians_score[min_score_idx])
						min_score_idx = k;
				}
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::renderCR(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA_CR << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::renderMark(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* visible_gaussians)
{
	renderCUDA_mark<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		visible_gaussians);
}