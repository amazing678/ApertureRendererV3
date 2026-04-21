// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "denoise/denoise.cuh"
#include "denoise/denoise_statistic.cuh"

#include "guiding/guiding.cuh"
#include "guiding/octahedron.cuh"
#include "render/object.cuh"
#include "object/sdf/sdf_cache_manager.cuh"

class SceneRenderer
{
private:
	// host
	int totalFrameIndex_ = 0;
	std::vector<SceneObject> sceneObjects_;
	BVHHost bvh_;

	// device
	SceneObject* sceneObjectsDevice_ = nullptr;
	AdditionalLightInfo* sceneLightsInfoDevice_ = nullptr;
	BVH bvhDevice_;

	// device, path guiding related
	Octahedron* probeDevice_ = nullptr;
	size_t probeCount_ = 0;
	Octahedron* probeTempDevice_ = nullptr; // for accumulate, temp, clear every frame
	//
	float3* denoisedHDRDevice_ = nullptr;
	size_t denoisedHDRDeviceAllocatedSize_ = 0;
public:
	size_t* probeIndirectIndexVolumeDevice_ = nullptr; // index points to probeDevice_
	
	int3 probeIndirectIndexVolumeStart_ = {0, 0, 0};// not on device
	int3 probeIndirectIndexVolumeEnd_ = {0, 0, 0};// not on device
	int3 probeIndirectIndexVolumeSize_ = {0, 0, 0};// not on device
public:
	// this variable not manage by scene renderer:
	pg::PathGuidingSample* sharedCollectedRadianceSampleDevice_ = nullptr;
	size_t maxRadianceSampleCount_ = 0;
	// screen gbuffer
	denoise::ScreenGBuffer* sharedScreenGBufferDevice_ = nullptr;
	// for reduce
	pg::PathGuidingSample* validCollectedRadianceSampleDevice_ = nullptr;
	size_t validRadianceSampleCount_ = 0;
	size_t validSampleCount_ = 0;
	// for unique
	int4* uniqueProbeVolumeDevice_ = nullptr;
	size_t uniqueVoxelCount_ = 0;
	// for unique & invalid
	int4* invalidUniqueVoxelDevice_ = nullptr;
	size_t invalidUniqueVoxelCount_ = 0;
	// for spectrum 
	float* spectrumRGBLutDevice_ = nullptr;
	float* spectrumLambdaLutDevice_ = nullptr;

	cudaStream_t producerStream_ = nullptr;
	cudaStream_t pathGuidingStream_ = nullptr;
	cudaEvent_t radianceSamplesReadyEvent_ = nullptr;
	cudaEvent_t volumeWrittenEvent_ = nullptr;
private:
	// state
	bool bSceneDirty_ = false;
	bool bNeedResetGuiding_ = true;
	
	SceneRenderer(const SceneRenderer& obj) = delete;
	void SetupSpectrumLUT();
	void SetupStream();
	void UpdateSceneProbe();
	void UpdateSceneToDeviceIfDirty();
	void FreeSceneRelatedDevicePtr();
	void FreeScreenRelatedDevicePtr();
	static void FreeIfNotNullptr(void** devicePtr)
	{
		if(*devicePtr)
		{
			cudaFree(*devicePtr);
			CHECK_ERROR();
		}
		*devicePtr = nullptr;
	}
	
public:
	SceneSetting sceneSetting_;
	SDFCacheManager* sdfManager_ = nullptr;
	//
	SceneRenderer(const std::vector<SceneObject>& sceneObjects);
	~SceneRenderer();

	void SetExposure(float exp);
	void MarkAsDirty()
	{
		bSceneDirty_ = true;
	}
	void MarkNeedResetGuiding()
	{
		bNeedResetGuiding_ = true;
	}

	std::vector<float3> Render(const int2 size, const float3 cameraOrigin, const float3 cameraDirection, int sampleNum = 1024);
	
	void Render(float3* hdrTarget, unsigned int* ldrTarget, const int2 size,
		const float3 cameraOrigin, const float3 cameraDirection,
		int frameIndex, int toneType,
		float3* oddTarget = nullptr, float3* evenTarget = nullptr,
		pg::PathGuidingSample* radianceSampleBuffer = nullptr,
		denoise::ScreenGBuffer* screenGBuffer = nullptr, denoise::ScreenStatisticsBuffer* screenStatisticsBuffer = nullptr);

	void UpdatePathGuidingSample();
	void ResetPathGuidingSample(const cudaStream_t stream);

	template <class SDFFunctor>
	inline void AddVolumeToCacheManager(int3 dims, float3 pMin, float3 pMax, const SDFFunctor& sdf, const SDFInfo& sdfInfo)
	{
		if(sdfManager_)
		{
			sdfManager_->AddVolume(dims, pMin, pMax, sdf, sdfInfo);
		}
	}

	inline void ClearCacheManager() const
	{
		if(sdfManager_)
		{
			sdfManager_->Clear();
		}
	}
};
