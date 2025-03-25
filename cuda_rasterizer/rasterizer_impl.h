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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* aabb_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	struct Timer
	{
		std::vector<std::string> names;
		std::vector<cudaEvent_t> start;
		std::vector<cudaEvent_t> end;
		float t;
		bool debug = false;

		Timer(): debug(false) {}
		Timer(bool DEBUG): debug(DEBUG) {} 
		
		void pushRange(const char* name)
		{
			if (debug)
			{
				names.push_back(name);
				start.push_back(cudaEvent_t());
				end.push_back(cudaEvent_t());
				cudaEventCreate(&start.back());
				cudaEventCreate(&end.back());

				cudaEventRecord(start.back());
			}
		}

		void pop()
		{
			if (debug)
			{
				cudaEventRecord(end.back());
				cudaEventSynchronize(end.back());
				cudaEventElapsedTime(&t, start.back(), end.back());
				printf("Execution time: %f ms, FPS: %f frames/sec\n", names.back().c_str(), t, 1000.0 / t);
				cudaEventDestroy(start.back());
				cudaEventDestroy(end.back());
				start.pop_back();
				end.pop_back();
				names.pop_back();
			}
		}
	};
};