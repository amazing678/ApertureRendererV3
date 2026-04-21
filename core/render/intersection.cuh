// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once

#include "vector.cuh"
#include "object.cuh"
#include "object/AABB.cuh"
#include "object/Sphere.cuh"
#include "object/sdf/sdf_master.cuh"
#include <cuda_runtime.h>

#include "object/sdf/sdf_torus.cuh"

namespace intersection
{
    // factory
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext IntersectBase(const int depth, float3 rayOrigin, float3 rayDirection, const SceneObject& object)
    {
        if(object.type_ == EObjectType::OBJ_CUBE)
        {
            return object::IntersectAABBObject(rayOrigin, rayDirection, object);
        }
        else if(object.type_ == EObjectType::OBJ_SPHERE)
        {
            return object::IntersectSphereObject(rayOrigin, rayDirection, object);
        }
        else if(object.type_ == EObjectType::OBJ_SDF)
        {
            return sdf::IntersectSDFObject(depth, rayOrigin, rayDirection, object);
        }
        return {};
    }

    // overlap factory
    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext OverlapBase(float3 rayOrigin, const SceneObject& object)
    {
        if(object.type_ == EObjectType::OBJ_CUBE)
        {
            return object::OverlapAABBObject(rayOrigin, object);
        }
        else if(object.type_ == EObjectType::OBJ_SPHERE)
        {
            return object::OverlapSphereObject(rayOrigin, object);
        }
        else if(object.type_ == EObjectType::OBJ_SDF)
        {
            return sdf::OverlapSDFObject(rayOrigin, object);
        }
        return {};
    }

    // intersector
    // [min, max)
    struct SceneObjectsIntersector
    {
        const SceneObject* sceneObjects_; // alias
        int sceneCount_ = -1;
    
        [[nodiscard]] FUNCTION_MODIFIER_INLINE bool operator()(
            const int depth,
            const int objectIndex,
            const float3 rayOrigin, const float3 rayDirection,
            const float tMin, const float tMax,
            IntersectionContext* __restrict__ outIntersectionContext = nullptr  // need full intersection info, fill all info here
        ) const
        {
            const SceneObject* __restrict__ restrictSceneObjects = sceneObjects_;
            const SceneObject& currentObject = restrictSceneObjects[objectIndex];
            const IntersectionContext intersectionContext = IntersectBase(depth, rayOrigin, rayDirection, currentObject);
            if (!intersectionContext.bHit_)
            {
                return false;
            }
            if (intersectionContext.distance_ < tMin || intersectionContext.distance_ >= tMax)
            {
                return false;
            }
            if (outIntersectionContext)
            {
                *outIntersectionContext = intersectionContext;
                outIntersectionContext->objectIndex_ = objectIndex;
                outIntersectionContext->lightIndex_ = currentObject.lightIndex_;
            }
            return true;
        }
    };

    // master
    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext Intersect(
        const int depth,
        const float3 rayOrigin,
        const float3 rayDirection,
        const Intersector& intersector,
        const int soloObjectIndex,
        float tMin,
        float tMax = inf())
    {
        if (soloObjectIndex >= 0)
        {
            IntersectionContext result{};
            if (intersector(depth, soloObjectIndex, rayOrigin, rayDirection, tMin, tMax, &result))
            {
                return result;
            }
            return result; // miss
        }
        IntersectionContext result{};
        result.distance_ = tMax;
        for (int i = 0; i < intersector.sceneCount_; ++i)
        {
            IntersectionContext hitResult{};
            if (intersector(depth, i, rayOrigin, rayDirection, tMin, result.distance_, &hitResult))
            {
                result = hitResult;
            }
        }
        return result;
    }

    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext AnyHitSegment(const int depth, const float3 sourcePoint, const float3 targetPoint, const Intersector& intersector)
    {
        float3 direction = targetPoint - sourcePoint;
        const float distance = length(direction);
        if (distance <= 0.0f)
        {
            return {};
        }
        direction = direction / distance;
        const float eps = INTERSECT_EPS(depth) * fmaxf(1.0f, fmaxf(length(sourcePoint), length(targetPoint)));
        const float tMin = eps;
        const float tMax = fmaxf(0.0f, distance - eps);
        if (tMax <= 0.0f)
        {
            return {};
        }
        for (int i = 0; i < intersector.sceneCount_; ++i)
        {
            IntersectionContext tmp{};
            if (intersector(depth, i, sourcePoint, direction, tMin, tMax, &tmp))
            {
                return tmp;
            }
        }
        return {};
    }

    template <bool bAnyHit, typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER IntersectionContext TraverseBVH(
        const int depth,
        const float3 rayOrigin,
        const float3 rayDirection,
        const Intersector& intersector,
        const BVHNode* __restrict__ nodes,
        const int nodeCount,
        const int* __restrict__ objectIndices,
        float tMin,
        float tMax = inf()
        )
    {
        IntersectionContext result = {};
        if (nodeCount == 0)
        {
            return result;
        }
        result.distance_ = tMax;

        constexpr int STACK_CAP = 128;
        int stack[STACK_CAP];
        int sp = 0;
        auto PUSH = [&](int n) 
        {
            if (sp < STACK_CAP)
            {
                stack[sp++] = n;
            }
#ifdef DEBUG
            else assert(false && "BVH traversal stack overflow");
#endif
        };
        PUSH(0);
        const float3 invDirection = invSafe(rayDirection);
        while (sp)
        {
            const int currentNodeIndex = stack[--sp];
            const BVHNode& currentNode = nodes[currentNodeIndex];
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
            const float4 prefetchMinLeftRight = currentNode.boundMinLeftRight_;
            const float4 prefetchMaxFirstCount = currentNode.boundMaxFirstCount_;
            const float3 currentNodeMin = xyz(prefetchMinLeftRight);
            const float3 currentNodeMax = xyz(prefetchMaxFirstCount);
            uint16_t currentNodeLeft;
            uint16_t currentNodeRight;
            uint16_t currentNodeFirst;
            uint16_t currentNodeCount;
            Unpack16x2(floatAsUint(prefetchMinLeftRight.w), currentNodeLeft, currentNodeRight);
            Unpack16x2(floatAsUint(prefetchMaxFirstCount.w), currentNodeFirst, currentNodeCount);
#else
            const float3 currentNodeMin = currentNode.boundMin_;
            const float3 currentNodeMax = currentNode.boundMax_;
            const int currentNodeLeft = currentNode.left_;
            const int currentNodeRight = currentNode.right_;
            const int currentNodeFirst = currentNode.first_;
            const int currentNodeCount = currentNode.count_;
#endif
            if (!FastAABBHit(rayOrigin, invDirection, currentNodeMin, currentNodeMax, tMin, result.distance_))
            {
                continue;
            }
            if (currentNodeCount > 0)
            {
                for (int i = 0; i < currentNodeCount; ++i)
                {
                    const int objectIndex = objectIndices[currentNodeFirst + i];
                    if (intersector(depth, objectIndex, rayOrigin, rayDirection, tMin, result.distance_, &result))
                    {
                        if constexpr (bAnyHit)
                        {
                            return result;
                        }
                    }
                }
            }
            else
            {
                const BVHNode& left = nodes[currentNodeLeft];
                const BVHNode& right = nodes[currentNodeRight];
#ifdef BVH_USE_SORT_STACK
                float tL = 0.0f;
                float tR = 0.0f;
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
                const bool hitL = FastAABBHitWithEnter(rayOrigin, invDirection, xyz(left.boundMinLeftRight_),  xyz(left.boundMaxFirstCount_),  tMin, result.distance_, tL);
                const bool hitR = FastAABBHitWithEnter(rayOrigin, invDirection, xyz(right.boundMinLeftRight_), xyz(right.boundMaxFirstCount_), tMin, result.distance_, tR);
#else
                const bool hitL = FastAABBHitWithEnter(rayOrigin, invDirection, left.boundMin_,  left.boundMax_,  tMin, result.distance_, tL);
                const bool hitR = FastAABBHitWithEnter(rayOrigin, invDirection, right.boundMin_, right.boundMax_, tMin, result.distance_, tR);
#endif
                if (hitL && hitR)
                {
                    if (tL > tR)
                    {
                        PUSH(currentNodeLeft);
                        PUSH(currentNodeRight);
                    }
                    else
                    {
                        PUSH(currentNodeRight);
                        PUSH(currentNodeLeft);
                    }
                }
                else if (hitL)
                {
                    PUSH(currentNodeLeft);
                }
                else if (hitR)
                {
                    PUSH(currentNodeRight);
                }
#else
#if defined(BVH_USE_16_BITS_NODE) && defined(BVH_PACK_NODE)
                const float3 leftCenter = AABBCenter(xyz(left.boundMinLeftRight_), xyz(left.boundMaxFirstCount_));
                const float3 rightCenter = AABBCenter(xyz(right.boundMinLeftRight_), xyz(right.boundMaxFirstCount_));
#else
                const float3 leftCenter = AABBCenter(left.boundMin_, left.boundMax_);
                const float3 rightCenter = AABBCenter(right.boundMin_, right.boundMax_);
#endif
                const float distanceLeft = dot(leftCenter  - rayOrigin,  rayDirection);
                const float distanceRight = dot(rightCenter - rayOrigin,  rayDirection);
                if (distanceLeft > distanceRight)
                {
                    PUSH(currentNodeLeft);
                    PUSH(currentNodeRight);
                }
                else
                {
                    PUSH(currentNodeRight);
                    PUSH(currentNodeLeft);
                }
#endif
            }
        }
        return result;
    }

    template <typename Intersector>
    [[nodiscard]] FUNCTION_MODIFIER_INLINE IntersectionContext AnyHitSegmentBVH(
        const int depth,
        const float3 sourcePoint, const float3 targetPoint, const Intersector& intersector,
        const BVHNode* __restrict__ nodes,
        const int nodeCount,
        const int* __restrict__ objectIndices)
    {
        float3 direction = targetPoint - sourcePoint;
        const float distance = length(direction);
        if (distance <= 0.0f)
        {
            return {};
        }
        direction = direction / distance;
        const float eps = INTERSECT_EPS(depth) * fmaxf(1.0f, fmaxf(length(sourcePoint), length(targetPoint)));
        const float tMin = 0.0f;
        const float tMax = fmaxf(0.0f, distance - eps);
        if (tMax <= 0.0f)
        {
            return {};
        }
        const IntersectionContext intersectionContext = TraverseBVH<true>( // AnyHit = true
            depth, sourcePoint, direction, intersector, nodes, nodeCount, objectIndices, tMin, tMax);
        return intersectionContext;
    }

    [[nodiscard]] FUNCTION_MODIFIER_INLINE OverlapContext Overlap(float3 rayOrigin, const SceneObject* __restrict__ sceneObjects, const int sceneObjectCounts)
    {
        OverlapContext result = {};
        for(int i = 0; i < sceneObjectCounts; i++)
        {
            const SceneObject& currentObject = sceneObjects[i];
            const OverlapContext overlapResult = OverlapBase(rayOrigin, currentObject);
            if (overlapResult.bOverlap_)
            {
                result = overlapResult;
                result.objectIndex_ = i;
                return result;
            }
        }
        return result;
    }
}
