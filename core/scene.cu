// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#include "scene.hpp"

#include <cassert>

#include "render/render.cuh"
#include "tonemap.cuh"
#include "config.cuh"

#include "platform.h"
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>

#include "denoise/denoise_statistic.cuh"
#include "spectrum/spectrum_lut.cuh"

__device__ float exposure = 1;

template<bool bDebugAlbedo, bool bDebugNormal, bool bDebugDepth, bool bDebugMetallic, bool bDebugRoughness,
        bool bDebugStatisticsNum, bool bDebugStatisticsMean, bool bDebugStatisticsM2, bool bDebugStatisticsM3>
__global__ void Tonemap(
    float3* __restrict__ target,
    const denoise::ScreenGBuffer* __restrict__ screenGBuffers,
    const denoise::ScreenStatisticsBuffer* __restrict__ screenStatisticsBuffers,
    unsigned int* __restrict__ target2, int2 size, int toneType)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size.x * size.y)
    {
        return;
    }
    if constexpr (bDebugAlbedo)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoX_)));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoY_)));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.albedoZ_)));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugNormal)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalX_) * 0.5f + 0.5f));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalY_) * 0.5f + 0.5f));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(static_cast<float>(currentGBuffer.normalZ_) * 0.5f + 0.5f));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugDepth)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(1.0f / (currentGBuffer.depth_ + 1.0f));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugMetallic)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(static_cast<float>(currentGBuffer.metallic_));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugRoughness)
    {
        const denoise::ScreenGBuffer& currentGBuffer = screenGBuffers[id];
        const float3 heat = color::HeatMapViridis(static_cast<float>(currentGBuffer.roughness_));
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsNum)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float heat = static_cast<float>(log(static_cast<double>(currentStatisticsBuffer.num_) + 1.0)) * 0.01f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat));
        target2[id] = 0xff000000 | (red << 16);
    }
    else if constexpr (bDebugStatisticsMean)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.mean_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsM2)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.M2DivNum_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else if constexpr (bDebugStatisticsM3)
    {
        const denoise::ScreenStatisticsBuffer& currentStatisticsBuffer = screenStatisticsBuffers[id];
        const float3 heat = abs(currentStatisticsBuffer.M3DivNum_) * 0.25f;
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(heat.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(heat.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(heat.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
    else
    {
#ifdef USE_SPECTRUM_RENDERING
        const float3 result = XYZ2SRGBLinearD65(target[id]);
#else
        const float3 result = target[id];
#endif
        float3 value = result * exposure;
        if (toneType == 1)
        {
            value = Gamma(value);
        }
        else if (toneType == 2)
        {
            value = ACES(value);
        }
        const unsigned int red = static_cast<unsigned int>(255.0f * saturate_(value.x));
        const unsigned int green = static_cast<unsigned int>(255.0f * saturate_(value.y));
        const unsigned int blue = static_cast<unsigned int>(255.0f * saturate_(value.z));
        target2[id] = 0xff000000 | (red << 16) | (green << 8) | blue;
    }
}

template<bool bUseBVH, bool bUseNEE, bool bUsePathGuiding>
__global__ void RenderCamera(float3* target, const int2 size, const float3 cameraOrigin, const float3 cameraDirection,
    float3* oddTarget = nullptr, float3* evenTarget = nullptr,
    pg::PathGuidingSample* radianceSampleBuffer = nullptr, denoise::ScreenGBuffer* screenGBuffer = nullptr, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer = nullptr) // radianceSampleBuffer must sized size.x * size.y * PATH_GUIDING_COLLECT_DEPTH
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // y
    if (i >= size.x || j >= size.y)
    {
        return;
    }
    const int idx = j * size.x + i;

    curandState seed;
    InitRand(&seed);
    float u = (i + Rand1(&seed)) / static_cast<float>(size.x);
    float v = (j + Rand1(&seed)) / static_cast<float>(size.y);

    float2 ndc = make_float2(u * 2.f - 1.f, (1.0 - v) * 2.f - 1.f);
    constexpr float FOV = 60.0f * 3.1415926f / 180.0f;
    const float aspect = static_cast<float>(size.x) / max(1.0f, static_cast<float>(size.y));
    const float tanHalfY  = tanf(0.5f * FOV);
    const float tanHalfX  = tanHalfY * aspect;

    const float sx = ndc.x * tanHalfX;
    const float sy = ndc.y * tanHalfY;

    const float3 right = normalize(cross(cameraDirection, float3{0.0, 1.0, 0.0}));
    const float3 up = normalize(cross(cameraDirection, right));
    const float3 rayDirection = normalize(cameraDirection + right * sx + up * sy);
    const float4 resultAndDistance = CalculateRadiance<bUseBVH, bUseNEE, bUsePathGuiding>(
        &seed, cameraOrigin, rayDirection,
        radianceSampleBuffer ? &radianceSampleBuffer[idx * PATH_GUIDING_COLLECT_DEPTH] : nullptr,
        screenGBuffer ? &screenGBuffer[idx] : nullptr,
        screenStatisticsBuffer ? &screenStatisticsBuffer[idx] : nullptr
    );

    float3 result = make_f3(resultAndDistance);
    result = max(float3{ 0 }, result);
    
    const float lerpRate = 1.0f / static_cast<float>(1 + FRAME_INDEX);
    target[idx] = lerp(target[idx], result, lerpRate);
    if(oddTarget != nullptr && evenTarget != nullptr)
    {
        const float lerpRateHalf = 1.0f / static_cast<float>(1 + (FRAME_INDEX >> 1));
        if(FRAME_INDEX % 2 == 1)
        {
            oddTarget[idx] = lerp(oddTarget[idx], result, lerpRateHalf);
        }
        else
        {
            evenTarget[idx] = lerp(evenTarget[idx], result, lerpRateHalf);
        }
    }
}

#define DISPATCH_RENDER_CAMERA(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer) do {\
        dim3 dimBlock(8, 4);\
        dim3 dimGrid;\
        dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;\
        dimGrid.y = (size.y + dimBlock.y - 1) / dimBlock.y;\
        cudaStreamWaitEvent(producerStream_, volumeWrittenEvent_, 0);\
        switch ((sceneSetting_.bUseBVH_?4:0) | (sceneSetting_.bUseNEE_?2:0) | (sceneSetting_.bUsePathGuiding_?1:0)) {\
            case 0: RenderCamera<false, false, false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 1: RenderCamera<false, false, true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 2: RenderCamera<false, true , false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 3: RenderCamera<false, true , true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 4: RenderCamera<true , false, false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 5: RenderCamera<true , false, true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 6: RenderCamera<true , true , false><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
            case 7: RenderCamera<true , true , true ><<<dimGrid, dimBlock, 0, producerStream_>>>(target, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer); break;\
        }\
        cudaEventRecord(radianceSamplesReadyEvent_, producerStream_);\
    }while(0)

std::vector<float3> SceneRenderer::Render(const int2 size, const float3 cameraOrigin, const float3 cameraDirection, int sampleNum) 
{
    SetupSpectrumLUT();
    UpdateSceneToDeviceIfDirty();
    
    float3* target;
    cudaMalloc(&target, size.x * size.y * sizeof(float3));

    for (int i = 0; i < sampleNum; i++)
    {
        cudaMemcpyToSymbol(FRAME_INDEX, &i, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(TOTAL_FRAME_INDEX, &i, sizeof(int), 0, cudaMemcpyHostToDevice);

        DISPATCH_RENDER_CAMERA(target, size, cameraOrigin, cameraDirection, nullptr, nullptr, nullptr, nullptr, nullptr);

        cudaDeviceSynchronize();

        CHECK_ERROR();

    }

    std::vector<float3> resultCPU(size.x * size.y);

    cudaMemcpy(resultCPU.data(), target, sizeof(float3) * size.x * size.y, cudaMemcpyDeviceToHost);
    cudaFree(target);
    CHECK_ERROR();

    return resultCPU;
}

void SceneRenderer::Render(
    float3* hdrTarget, unsigned int* ldrTarget, const int2 size,
    const float3 cameraOrigin, const float3 cameraDirection, int frameIndex, int toneType,
    float3* oddTarget, float3* evenTarget, pg::PathGuidingSample* radianceSampleBuffer,
    denoise::ScreenGBuffer* screenGBuffer, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer) 
{
    SetupSpectrumLUT();
    UpdateSceneProbe();   
    UpdateSceneToDeviceIfDirty();
    cudaMemcpyToSymbol(FRAME_INDEX, &frameIndex, sizeof(int), 0, cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpyToSymbol(TOTAL_FRAME_INDEX, &totalFrameIndex_, sizeof(int), 0, cudaMemcpyHostToDevice);    
    CHECK_ERROR();

    totalFrameIndex_++;
    //
    DISPATCH_RENDER_CAMERA(hdrTarget, size, cameraOrigin, cameraDirection, oddTarget, evenTarget, radianceSampleBuffer, screenGBuffer, screenStatisticsBuffer);
    CHECK_ERROR();
    if(sceneSetting_.bUsePathGuiding_)
    {
        UpdatePathGuidingSample();
        CHECK_ERROR();
    }
    else
    {
        const cudaStream_t stream = pathGuidingStream_;
        cudaStreamWaitEvent(pathGuidingStream_, radianceSamplesReadyEvent_, 0);
        CHECK_ERROR();
        // reset everything related to path guiding
        ResetPathGuidingSample(stream);
        CHECK_ERROR();
        cudaEventRecord(volumeWrittenEvent_, stream);
        CHECK_ERROR();
    }
    //
    int taskNum = size.x * size.y;
    int group = 32;
    int group_num = taskNum / group + (taskNum % group != 0 ? 1 : 0);
#define CALL_OUTPUT_FUNCTION(hdrTarget) <<<group_num, group>>>(hdrTarget, screenGBuffer, screenStatisticsBuffer, ldrTarget, size, toneType);
    if(sceneSetting_.bDebugAlbedo_)
    {
        Tonemap<true, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugNormal_)
    {
        Tonemap<false, true, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugDepth_)
    {
        Tonemap<false, false, true, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugMetallic_)
    {
        Tonemap<false, false, false, true, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugRoughness_)
    {
        Tonemap<false, false, false, false, true, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsNum_)
    {
        Tonemap<false, false, false, false, false, true, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsMean_)
    {
        Tonemap<false, false, false, false, false, false, true, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsM2_)
    {
        Tonemap<false, false, false, false, false, false, false, true, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDebugStatisticsM3_)
    {
        Tonemap<false, false, false, false, false, false, false, false, true>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
    else if(sceneSetting_.bDenoise_)
    {
        const int numel = size.x * size.y;
        if(denoisedHDRDevice_ == nullptr || denoisedHDRDeviceAllocatedSize_ != numel)
        {
            if(denoisedHDRDevice_)
            {
                cudaFree(denoisedHDRDevice_);
                CHECK_ERROR();
            }
            cudaMalloc(&denoisedHDRDevice_, sizeof(float3) * numel);
            CHECK_ERROR();
            denoisedHDRDeviceAllocatedSize_ = numel;
        }
        const dim3 blockBilateral(16, 16);
        const dim3 gridBilateral( (size.x + blockBilateral.x - 1) / blockBilateral.x,
                                  (size.y + blockBilateral.y - 1) / blockBilateral.y );
        denoise::Bilateral16x16<7><<<gridBilateral, blockBilateral>>>(
            hdrTarget,
            denoisedHDRDevice_,
            screenGBuffer,
            screenStatisticsBuffer,
            size
        );
        CHECK_ERROR();
        Tonemap<false, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(denoisedHDRDevice_)
        CHECK_ERROR();
    }
    else
    {
        Tonemap<false, false, false, false, false, false, false, false, false>CALL_OUTPUT_FUNCTION(hdrTarget)
        CHECK_ERROR();
    }
#undef CALL_OUTPUT_FUNCTION
    //
    CHECK_ERROR();

    cudaDeviceSynchronize();
    CHECK_ERROR();

    CHECK_ERROR();
    return;
}

void SceneRenderer::ResetPathGuidingSample(const cudaStream_t stream)
{
    if (sharedCollectedRadianceSampleDevice_ && maxRadianceSampleCount_ > 0)
    {
        cudaMemsetAsync(sharedCollectedRadianceSampleDevice_, 0, maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample), pathGuidingStream_);
        CHECK_ERROR();
    }
    FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeDevice_));
    CHECK_ERROR();

    validRadianceSampleCount_ = 0;
    validSampleCount_ = 0;
    uniqueVoxelCount_ = 0;
    invalidUniqueVoxelCount_ = 0;
    probeCount_ = 0;
    probeIndirectIndexVolumeStart_ = make_int3(0,0,0);
    probeIndirectIndexVolumeEnd_ = make_int3(0,0,0);
    probeIndirectIndexVolumeSize_ = make_int3(0,0,0);
    //
    Octahedron* nullProbe = nullptr;
    size_t* nullIndex = nullptr;
    size_t zeroSizeT = 0;
    int3 zero3 = make_int3(0,0,0);
    // Probes
    cudaMemcpyToSymbolAsync(SCENE_PROBES, &nullProbe, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(SCENE_PROBES_COUNTS, &zeroSizeT, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR();

    // Indirect index volume + bounds
    cudaMemcpyToSymbolAsync(INDIRECT_INDEX_VOLUME, &nullIndex, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_START, &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_END,   &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_SIZE,  &zero3, sizeof(int3), 0, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR();
}

void SceneRenderer::UpdatePathGuidingSample()
{
    const cudaStream_t stream = pathGuidingStream_;
    cudaStreamWaitEvent(pathGuidingStream_, radianceSamplesReadyEvent_, 0);
    CHECK_CUDA_SYNC();
    if(bNeedResetGuiding_)
    {
        ResetPathGuidingSample(stream);
        bNeedResetGuiding_ = false;
    }
    CHECK_CUDA_SYNC();
    if(validRadianceSampleCount_ != maxRadianceSampleCount_ || validCollectedRadianceSampleDevice_ == nullptr)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
        CHECK_CUDA_SYNC();
        FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
        CHECK_CUDA_SYNC();
        // reallocate
        cudaMalloc(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_), maxRadianceSampleCount_ * sizeof(pg::PathGuidingSample));
        validRadianceSampleCount_ = maxRadianceSampleCount_;
        cudaMalloc(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_), maxRadianceSampleCount_ * sizeof(int4));
    }
    validSampleCount_ = pg::FilterValidSamples(sharedCollectedRadianceSampleDevice_, maxRadianceSampleCount_, validCollectedRadianceSampleDevice_, stream);
    CHECK_ERROR();
    //printf("validSampleCount_ vs totalSampleCount: %zd, %zd\n", validSampleCount_, maxRadianceSampleCount_);
    uniqueVoxelCount_ = pg::UniqueVoxelGridsFromValid(validCollectedRadianceSampleDevice_, validSampleCount_, uniqueProbeVolumeDevice_, stream);
    CHECK_ERROR();
    if(uniqueVoxelCount_ <= 0)
    {
        return;
    }
    //printf("uniqueVoxelCount_ vs validSampleCount_: %zd, %zd\n", uniqueVoxelCount_, validSampleCount_);
    const auto [voxelMin, voxelMax] = pg::GetMinMax(uniqueProbeVolumeDevice_, uniqueVoxelCount_, stream);
    CHECK_ERROR();
    //printf("voxel min: %d, %d, %d; voxel max: %d, %d, %d\n", voxelMin.x, voxelMin.y, voxelMin.z, voxelMax.x, voxelMax.y, voxelMax.z);
    const int3 desiredVolumeStart = {
        min(probeIndirectIndexVolumeStart_.x, voxelMin.x),
        min(probeIndirectIndexVolumeStart_.y, voxelMin.y),
        min(probeIndirectIndexVolumeStart_.z, voxelMin.z)
    };
    const int3 desiredVolumeEnd = {
        max(probeIndirectIndexVolumeEnd_.x, voxelMax.x),
        max(probeIndirectIndexVolumeEnd_.y, voxelMax.y),
        max(probeIndirectIndexVolumeEnd_.z, voxelMax.z)
    };
    if(probeIndirectIndexVolumeStart_.x != desiredVolumeStart.x || probeIndirectIndexVolumeStart_.y != desiredVolumeStart.y || probeIndirectIndexVolumeStart_.z != desiredVolumeStart.z ||
       probeIndirectIndexVolumeEnd_.x != desiredVolumeEnd.x || probeIndirectIndexVolumeEnd_.y != desiredVolumeEnd.y || probeIndirectIndexVolumeEnd_.z != desiredVolumeEnd.z)
    {
        const int3 newDim = pg::DimsFromBoundsInclusive(desiredVolumeStart, desiredVolumeEnd);
        CHECK_ERROR();
        const size_t voxels = static_cast<size_t>(newDim.x) * newDim.y * newDim.z;
        const size_t elements = voxels * PATH_GUIDING_FACE_COUNT;
        // need to reallocate
        if(probeIndirectIndexVolumeSize_.x == 0 || probeIndirectIndexVolumeSize_.y == 0 || probeIndirectIndexVolumeSize_.z == 0)
        {
            // allocate new memory
            cudaMalloc(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_), elements * sizeof(size_t));
            CHECK_ERROR();
            thrust::fill_n(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(probeIndirectIndexVolumeDevice_),
                           elements, SIZE_T_MAX);
            CHECK_ERROR();
            //printf("Allocated New Indirect Volume: %d, %d, %d.\n", newDim.x, newDim.y, newDim.z);
        }
        else
        {
            size_t* probeIndirectIndexVolumeDeviceOld = probeIndirectIndexVolumeDevice_;
            probeIndirectIndexVolumeDevice_ = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_), elements * sizeof(size_t));
            CHECK_ERROR();
            pg::CopyOldIntoNew(probeIndirectIndexVolumeDeviceOld,
                                      probeIndirectIndexVolumeStart_, probeIndirectIndexVolumeEnd_,
                                      probeIndirectIndexVolumeDevice_,
                                      desiredVolumeStart, desiredVolumeEnd,
                                      SIZE_T_MAX, stream); // default value
            CHECK_ERROR();
            FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDeviceOld));
            CHECK_ERROR();
            //printf("Expand Indirect Volume From %d, %d, %d To %d, %d, %d.\n", probeIndirectIndexVolumeSize_.x, probeIndirectIndexVolumeSize_.y, probeIndirectIndexVolumeSize_.z, newDim.x, newDim.y, newDim.z);
        }
        cudaMemcpyToSymbolAsync(INDIRECT_INDEX_VOLUME, &probeIndirectIndexVolumeDevice_, sizeof(size_t*), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        probeIndirectIndexVolumeStart_ = desiredVolumeStart;
        probeIndirectIndexVolumeEnd_ = desiredVolumeEnd;
        probeIndirectIndexVolumeSize_ = newDim;
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_START, &probeIndirectIndexVolumeStart_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_END, &probeIndirectIndexVolumeEnd_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        cudaMemcpyToSymbolAsync(INDIRECT_VOLUME_SIZE, &probeIndirectIndexVolumeSize_, sizeof(int3), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        CHECK_ERROR();
        // update invalidUniqueVoxelDevice too in case not enough capacity
        FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
        cudaMalloc(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_), elements * sizeof(int4));
        CHECK_ERROR();
    }
    // filter invalid indirect voxel from uniqueProbeVolumeDevice_, uniqueVoxelCount_, probeIndirectIndexVolumeDevice_
    invalidUniqueVoxelCount_ = pg::FilterUnallocatedGrids(
                                uniqueProbeVolumeDevice_,
                                uniqueVoxelCount_,
                                probeIndirectIndexVolumeDevice_,
                                probeIndirectIndexVolumeStart_,
                                probeIndirectIndexVolumeSize_,
                                invalidUniqueVoxelDevice_,
                                stream);
    CHECK_ERROR();
    //printf("uniqueVoxelCount_ vs invalidUniqueVoxelCount_: %zd, %zd\n", uniqueVoxelCount_, invalidUniqueVoxelCount_);
    // alloc new probe
    if(invalidUniqueVoxelCount_ > 0)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
        Octahedron* probeDeviceOld = probeDevice_;
        const size_t newProbeCount = (probeCount_ + invalidUniqueVoxelCount_);
        cudaMalloc(reinterpret_cast<void**>(&probeDevice_), newProbeCount * sizeof(Octahedron));
        cudaMalloc(reinterpret_cast<void**>(&probeTempDevice_), newProbeCount * sizeof(Octahedron)); // temp
        cudaMemsetAsync(reinterpret_cast<char*>(probeDevice_) + probeCount_ * sizeof(Octahedron), 0, invalidUniqueVoxelCount_ * sizeof(Octahedron), stream);
        cudaMemsetAsync(reinterpret_cast<char*>(probeTempDevice_), 0, newProbeCount * sizeof(Octahedron), stream); // temp, all set to 0
        if (probeDeviceOld && probeCount_ > 0)
        {
            cudaMemcpy(probeDevice_, probeDeviceOld,
                       probeCount_ * sizeof(Octahedron),
                       cudaMemcpyDeviceToDevice);
            CHECK_ERROR();
        }
        CHECK_ERROR();
        FreeIfNotNullptr(reinterpret_cast<void**>(&probeDeviceOld));
        //
        pg::ScatterAssignIndices(invalidUniqueVoxelDevice_,
                                invalidUniqueVoxelCount_,
                                probeIndirectIndexVolumeDevice_,
                                probeIndirectIndexVolumeStart_,
                                probeIndirectIndexVolumeSize_,
                                probeCount_,
                                stream);
        probeCount_ = probeCount_ + invalidUniqueVoxelCount_;
        cudaMemcpyToSymbolAsync(SCENE_PROBES, &probeDevice_, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        cudaMemcpyToSymbolAsync(SCENE_PROBES_COUNTS, &probeCount_, sizeof(size_t), 0, cudaMemcpyHostToDevice, pathGuidingStream_);
        //printf("Write New Indirect Index Volume Count %zd: \n", invalidUniqueVoxelCount_);
        int4* invalidUniqueVoxelCPU = new int4[invalidUniqueVoxelCount_];
        cudaMemcpy(invalidUniqueVoxelCPU, invalidUniqueVoxelDevice_, sizeof(int4) * invalidUniqueVoxelCount_, cudaMemcpyDeviceToHost);
        CHECK_ERROR();
        //for(int i=0;i<invalidUniqueVoxelCount_;i++)
        {
            //printf("%d, %d, %d, %d\n", invalidUniqueVoxelCPU[i].x, invalidUniqueVoxelCPU[i].y, invalidUniqueVoxelCPU[i].z, invalidUniqueVoxelCPU[i].w);
        }
    }
    // write validCollectedRadianceSampleDevice_, validSampleCount_ to probeDeviceTemp_,
    if(validSampleCount_ > 0)
    {
        pg::AccumulateSamplesToOctaLeaf(validCollectedRadianceSampleDevice_, validSampleCount_,
                                                probeIndirectIndexVolumeDevice_, probeIndirectIndexVolumeStart_, probeIndirectIndexVolumeSize_,
                                                probeTempDevice_,
                                                stream);
        CHECK_ERROR();
        // merge all, TODO: merge those only updated
        pg::LaunchMergeProbeKernel(probeDevice_, probeTempDevice_, probeCount_, stream);
        CHECK_ERROR();
        cudaMemset(reinterpret_cast<char*>(probeTempDevice_), 0, probeCount_ * sizeof(Octahedron)); // temp, all set to 0
        CHECK_ERROR();
    }
    cudaEventRecord(volumeWrittenEvent_, stream);
}

void SceneRenderer::FreeSceneRelatedDevicePtr()
{
    FreeIfNotNullptr(reinterpret_cast<void**>(&bvhDevice_.objectIndices_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&bvhDevice_.nodes_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&sceneObjectsDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&sceneLightsInfoDevice_));
    // dont free screen size related device ptr like probeDevice_, collectedRadianceSampleDevice_
}

void SceneRenderer::FreeScreenRelatedDevicePtr()
{
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeTempDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&validCollectedRadianceSampleDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&probeIndirectIndexVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&uniqueProbeVolumeDevice_));
    FreeIfNotNullptr(reinterpret_cast<void**>(&invalidUniqueVoxelDevice_));
}

void SceneRenderer::SetupSpectrumLUT()
{
#ifdef USE_SPECTRUM_RENDERING
    if(spectrumRGBLutDevice_ == nullptr || spectrumLambdaLutDevice_ == nullptr)
    {
        FreeIfNotNullptr(reinterpret_cast<void**>(&spectrumRGBLutDevice_));
        FreeIfNotNullptr(reinterpret_cast<void**>(&spectrumLambdaLutDevice_));
        constexpr size_t rgbLutSize = SPECTRUM_RGB_LUT_RES * SPECTRUM_RGB_LUT_RES * SPECTRUM_RGB_LUT_RES * spectrum::query::KERNEL * sizeof(float);
        constexpr size_t lambdaLutSize = SPECTRUM_LAMBDA_LUT_RES * spectrum::query::KERNEL * sizeof(float);
        cudaMalloc(reinterpret_cast<void**>(&spectrumRGBLutDevice_), rgbLutSize);
        cudaMalloc(reinterpret_cast<void**>(&spectrumLambdaLutDevice_), lambdaLutSize);
        spectrum::PrecomputeSpectrumLUTsToDevice(spectrumRGBLutDevice_, spectrumLambdaLutDevice_);
        cudaMemcpyToSymbol(SPECTRUM_LUT_RGB,    &spectrumRGBLutDevice_,    sizeof(spectrumRGBLutDevice_));
        cudaMemcpyToSymbol(SPECTRUM_LUT_LAMBDA, &spectrumLambdaLutDevice_, sizeof(spectrumLambdaLutDevice_));
    }
#endif
}

void SceneRenderer::SetupStream()
{
    if(pathGuidingStream_ == nullptr)
    {
        cudaStreamCreateWithFlags(&producerStream_, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&pathGuidingStream_, cudaStreamNonBlocking);
        
        cudaEventCreateWithFlags(&radianceSamplesReadyEvent_,  cudaEventDisableTiming);
        cudaEventCreateWithFlags(&volumeWrittenEvent_, cudaEventDisableTiming);
        cudaEventRecord(volumeWrittenEvent_, pathGuidingStream_);
    }
}

void SceneRenderer::UpdateSceneProbe()
{
    //if(true)
    /*
    if(bSceneDirty_)
    {
        Octahedron testProbe;
        for(uint16_t x=0u;x< Octahedron::resolution_;x++)
        {
            for(uint16_t y=0u;y< Octahedron::resolution_;y++)
            {
#ifdef USE_3_CHANNEL_PROBE
                testProbe(x, y) = float3{
                    Hash11f(static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 16)) * 0.5f + 0.5f,
                    Hash11f((static_cast<uint32_t>(x) + 1314) | ((static_cast<uint32_t>(y) + 1314) << 16)) * 0.5f + 0.5f,
                    Hash11f((static_cast<uint32_t>(x) + 3756314) | ((static_cast<uint32_t>(y) + 3756314) << 16)) * 0.5f + 0.5f
                    };
#else
                testProbe(x, y) = Hash11f(static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 16)) * 0.5f + 0.5f;
#endif
            }
        }
        testProbe.GenerateMipmaps();
        cudaMalloc(reinterpret_cast<void**>(&probeDevice_), 1 * sizeof(Octahedron));
        cudaMemcpy(probeDevice_, &testProbe, 1 * sizeof(Octahedron), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_PROBES, &probeDevice_, sizeof(Octahedron*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        constexpr int probeCount = 1;
        cudaMemcpyToSymbol(SCENE_PROBES_COUNTS, &probeCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
    }
    */
}

void SceneRenderer::UpdateSceneToDeviceIfDirty()
{
    if(bSceneDirty_)
    {
        CHECK_ERROR();
        // update scene setting first
        cudaMemcpyToSymbol(SETTING, &sceneSetting_, sizeof(SceneSetting), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        // release old resource
        FreeSceneRelatedDevicePtr();
        CHECK_ERROR();
        // update bvh
        std::vector<PrimitiveProxy> proxys;
        proxys.reserve(sceneObjects_.size());
        for (auto& sceneObject : sceneObjects_)
        {
            proxys.push_back(sceneObject.proxy_);
        }
        bvh_ = BuildBVH(proxys);
        // filter out lights
        std::vector<AdditionalLightInfo> lightsInfo;
        float totalLightWeightSum = 0.0f;
        for(int i=0;i<sceneObjects_.size();i++)
        {
            const auto& currentObject = sceneObjects_[i];
            sceneObjects_[i].objectIndex_ = i; // self map
            if(sceneObjects_[i].material_.shadingModel_ == EShadingModel::MAT_LIGHT)
            {
                sceneObjects_[i].lightIndex_ = static_cast<int>(lightsInfo.size()); // light map
                
                AdditionalLightInfo currentInfo;
                currentInfo.objectIndex_ = i;
                currentInfo.importanceWeight_ = fmaxf(currentObject.GetArea() * currentObject.GetEmissivePower(), 1.0e-20f);
                totalLightWeightSum += currentInfo.importanceWeight_;
                lightsInfo.push_back(currentInfo);
            }
            else
            {
                sceneObjects_[i].lightIndex_ = -1;
            }
        }
        for (auto& lightInfo : lightsInfo)
        {
            lightInfo.selectProb_ = lightInfo.importanceWeight_ / totalLightWeightSum;
        }
        cudaMalloc(reinterpret_cast<void**>(&sceneLightsInfoDevice_), lightsInfo.size() * sizeof(AdditionalLightInfo));
        cudaMemcpy(sceneLightsInfoDevice_, lightsInfo.data(), lightsInfo.size() * sizeof(AdditionalLightInfo), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_LIGHTS_INFO, &sceneLightsInfoDevice_, sizeof(AdditionalLightInfo*), 0, cudaMemcpyHostToDevice);
        const int lightCount = static_cast<int>(lightsInfo.size());
        cudaMemcpyToSymbol(SCENE_LIGHTS_COUNTS, &lightCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        
        // copy bvh to device
        cudaMalloc(reinterpret_cast<void**>(&bvhDevice_.nodes_), bvh_.nodes_.size() * sizeof(BVHNode));
        cudaMemcpy(bvhDevice_.nodes_, bvh_.nodes_.data(), bvh_.nodes_.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(BVH_NODES, &bvhDevice_.nodes_, sizeof(BVHNode*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
	    const int nodeCount = static_cast<int>(bvh_.nodes_.size());
        cudaMemcpyToSymbol(BVH_NODES_COUNTS, &nodeCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        cudaMalloc(reinterpret_cast<void**>(&bvhDevice_.objectIndices_), bvh_.objectIndices_.size() * sizeof(int));
        cudaMemcpy(bvhDevice_.objectIndices_, bvh_.objectIndices_.data(), bvh_.objectIndices_.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(BVH_OBJECT_INDICES, &bvhDevice_.objectIndices_, sizeof(int*), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        assert(bvh_.objectIndices_.size() == sceneObjects_.size());
        //
        cudaMalloc(reinterpret_cast<void**>(&sceneObjectsDevice_), sceneObjects_.size() * sizeof(SceneObject));
        cudaMemcpy(sceneObjectsDevice_, sceneObjects_.data(), sceneObjects_.size() * sizeof(SceneObject), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(SCENE_OBJECTS, &sceneObjectsDevice_, sizeof(SceneObject*), 0, cudaMemcpyHostToDevice);
	    const int objectCount = static_cast<int>(sceneObjects_.size());
        cudaMemcpyToSymbol(SCENE_OBJECT_COUNTS, &objectCount, sizeof(int), 0, cudaMemcpyHostToDevice);
        CHECK_ERROR();
        
        bSceneDirty_ = false;
    }
}

SceneRenderer::SceneRenderer(const std::vector<SceneObject>& sceneObjects)
{
    SetupStream();
    sdfManager_ = new SDFCacheManager(producerStream_);
    sceneObjects_ = sceneObjects; // copy
    CHECK_ERROR();
    MarkAsDirty();
    printf("Scene Loaded \n");
    printf("Memory allocated \n");
}

SceneRenderer::~SceneRenderer() 
{
    delete sdfManager_;
    FreeSceneRelatedDevicePtr();
    FreeScreenRelatedDevicePtr();
}

void SceneRenderer::SetExposure(float exp)
{
    cudaMemcpyToSymbol(exposure, &exp, sizeof(float), 0, cudaMemcpyHostToDevice);
}