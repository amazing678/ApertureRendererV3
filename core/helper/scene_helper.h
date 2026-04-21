// Copyright (c) 2025 Yu Chengzhong <yuchengzhongUE4@gmail.com>
#pragma once
#include "random.cuh"
#include "scene.hpp"
#include "render/color.cuh"

struct SceneHelper
{
    constexpr static inline float THICK = 0.01f;
    inline static std::vector<SceneObject> CreateCornellBox(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float3 wallAColor = color::Red(),
        const float3 wallBColor = color::Green(),
        const float3 lightColor = color::White() * 10.0f,
        const float lightSize = 0.25f, const float3 lightOffset = {0.0f, 0.0f, 0.0f}, const bool multipleLight = false)
    {
        std::vector<SceneObject> Scene;
        // down
        Scene.emplace_back(
            float3{ 0.0f,-0.5f - THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.25f), 0.05f, 0.0f, 0.5f), // Color::WhiteDarken(0.25f)
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // up
        Scene.emplace_back(
            float3{ 0.0f,0.5f + THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // light
        if(multipleLight)
        {
            Scene.emplace_back(
                float3{ 0.25f, 0.5f - THICK, 0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ -0.25f, 0.5f - THICK, 0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ 0.25f, 0.5f - THICK, -0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            Scene.emplace_back(
                float3{ -0.25f, 0.5f - THICK, -0.25f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor * 0.25f),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
            
        }
        else
        {
            Scene.emplace_back(
                float3{ 0.0f,0.5f - THICK,0.0f } + position + lightOffset, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
                Material::CreateLight(lightColor),
                float3{0.0f, 0.0f, 0.0f},
                EObjectType::OBJ_CUBE
            );
        }
        // back
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.5f - THICK } + position, float3{ 0.5f,0.5f,THICK },
            Material::CreateGGXPureColor(color::WhiteDarken(0.25f), 0.25f, 1.0f, 0.5f),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );

        Scene.emplace_back(
            float3{ -0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallAColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallBColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    inline static std::vector<SceneObject> CreateDiffuseCornellBox(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float3 wallAColor = color::Red(),
        const float3 wallBColor = color::Green(),
        const float3 lightColor = color::White() * 10.0f,
        const float lightSize = 0.25f)
    {
        std::vector<SceneObject> Scene;
        // down
        Scene.emplace_back(
            float3{ 0.0f,-0.5f - THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // up
        Scene.emplace_back(
            float3{ 0.0f,0.5f + THICK,0.0f } + position, float3{ 0.5f,THICK,0.5f },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // light
        Scene.emplace_back(
            float3{ 0.0f,0.5f - THICK,0.0f } + position, float3{ 0.125f * lightSize,THICK,0.125f * lightSize},
            Material::CreateLight(lightColor),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // back
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.5f - THICK } + position, float3{ 0.5f,0.5f,THICK },
            Material::CreateDiffusePureColor(color::White()),
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );

        Scene.emplace_back(
            float3{ -0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallAColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.5f,0.0f,0.0f } + position, float3{ THICK,0.5f,0.5f },
            Material::CreateDiffusePureColor(wallBColor), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateSpectrumTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f})
    {
        std::vector<SceneObject> Scene;
        constexpr int WIDTH = 6;
        constexpr int WIDTH_HALF = WIDTH / 2;
        constexpr float SCALE = 0.6f;
        for(int x=-WIDTH_HALF; x<=WIDTH_HALF; x++)
        {
            for(int y=-WIDTH_HALF; y<=WIDTH_HALF; y++)
            {
                for(int z=-WIDTH_HALF; z<=WIDTH_HALF; z++)
                {
                    const float3 center = float3{(static_cast<float>(x) + static_cast<float>(z) / WIDTH) / WIDTH_HALF, (static_cast<float>(y) + static_cast<float>(x) / WIDTH) / WIDTH_HALF, static_cast<float>(z) / WIDTH_HALF} * 0.5f * SCALE;
                    const float abbe = lerp(50.0f, 1.0f, (static_cast<float>(z + WIDTH_HALF) / WIDTH));
                    const float tilt = lerp(1.0f, -1.0f, (static_cast<float>(x + WIDTH_HALF) / WIDTH));
                    const float ior = lerp(0.0f, 1.0f, (static_cast<float>(y + WIDTH_HALF) / WIDTH));
                    Scene.emplace_back(
                        center + position, float3{ 0.045f,0.1f,0.045f } * SCALE * 4 / WIDTH,
                        Material::CreateGlassPureColor(color::WhiteDarken(0.5f), lerp(1.3f, 2.5f, ior), 10.0f, abbe, tilt), 
                        float3{0.0f, 0.0f, 0.0f},
                        EObjectType::OBJ_CUBE
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0f, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass ball 1
        Scene.emplace_back(
            float3{ 0.0f,0.25f,0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.6f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // test light
        Scene.emplace_back(
            float3{ 0.125f,0.25f,-0.125f } + position, float3{ 0.05f,0.05f,0.05f },
            Material::CreateLight(color::White() * 25.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 2
        Scene.emplace_back(
            float3{ -0.25f,0.0f,-0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::Yellow(), lerp(1.0f, 1.5f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 3
        Scene.emplace_back(
            float3{ 0.25f,0.0f,0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::Cyan(), lerp(1.0, 1.55f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 4
        Scene.emplace_back(
            float3{ 0.0,0.0,0.0 } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::RedLighten(0.01f), lerp(1.0f, 1.45f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass ball 5
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,-0.125f } + position, float3{ 0.125f,0.125f,0.125f },
            Material::CreateGlassPureColor(color::BlueLighten(0.01f), lerp(1.0f, 1.5f, IORStrength), 1.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass plane 1
        Scene.emplace_back(
            float3{ 0.0f,0.0f,0.25f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::Azure(), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // glass plane 2
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::RedLighten(0.25f), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // glass plane 3
        Scene.emplace_back(
            float3{ 0.0f,0.25f,0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.25f), lerp(1.0f, 1.5f, IORStrength), 25.0f, abbeNumber, tiltNumber), 
            float3{-30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 1
        Scene.emplace_back(
            float3{ 0.0f,0.0f,-0.25f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::Orange(), 0.1f, 1.0f, 1.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 2
        Scene.emplace_back(
            float3{ 0.0f,-0.25f,-0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.9f), 0.1f, 0.0f, 0.5f), 
            float3{-30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        // ggx plane 3
        Scene.emplace_back(
            float3{ 0.0f,0.25f,-0.3f } + position, float3{ 0.125f,0.125f,0.01f },
            Material::CreateGGXPureColor(color::WhiteDarken(0.5f), 0.5f, 1.0f, 1.0f), 
            float3{30.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateBigGlassTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0)
    {
        std::vector<SceneObject> Scene;
        // test light
        /*
        Scene.emplace_back(
            float3{ 0.125,0.25,-0.125 } + position, float3{ 0.05f,0.05f,0.05f },
            Material::CreateLight(Color::White() * 25.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        */
        // glass ball
        Scene.emplace_back(
            float3{ -0.15f,0.0f,-0.2f } + position, float3{ 0.25f,0.25f,0.25f },
            Material::CreateGlassPureColor(color::CyanLighten(0.2f), lerp(1.0f, 1.45f, IORStrength), 5.0f), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass cube
        Scene.emplace_back(
            float3{ 0.15f,-0.2f,0.15f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.2f), lerp(1.0f, 1.45f, IORStrength), 5.0f), 
            float3{15.0f, 30.0f, 70.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateCubeTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass cube
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::PurpleLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ -0.2f,-0.2f,0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ -0.2f,-0.2f,-0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::CyanLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,-0.2f } + position, float3{ 0.075f,0.2f,0.075f },
            Material::CreateGlassPureColor(color::YellowLighten(0.5f), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateAbbeTestScene(const float3 position = float3{0.0f, 0.0f, 0.0f}, const float IORStrength = 1.0, const float abbeNumber = 20.0, const float tiltNumber = 0.0f)
    {
        std::vector<SceneObject> Scene;
        // glass sphere
        Scene.emplace_back(
            float3{ 0.2f,-0.2f,0.1f } + position, float3{ 0.2f,0.2f,0.2f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{0.0f, 0.0f, 0.0f},
            EObjectType::OBJ_SPHERE
        );
        // glass cube
        Scene.emplace_back(
            float3{ -0.1f,0.1f,-0.15f } + position, float3{ 0.1f,0.25f,0.1f },
            Material::CreateGlassPureColor(color::White(), lerp(1.0f, 1.45f, IORStrength), 5.0f, abbeNumber, tiltNumber), 
            float3{15.0f, 30.0f, 70.0f},
            EObjectType::OBJ_CUBE
        );
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTestScene2(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2,
        const float3 Color = color::Red(),
        const EObjectType objectType = EObjectType::OBJ_SPHERE,
        const float randomDiffuseRate = 0.0f)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const auto currentColor = lerp(Color, color::White(), normalized.x);
                    Scene.emplace_back(
                        location, float3{ 0.025f,0.025f,0.025f } * scale * objectScale,
                        random < randomDiffuseRate ? Material::CreateDiffusePureColor(currentColor) : Material::CreateGlassPureColor(currentColor, 1.1f + normalized.y, 1.0f + normalized.z * 100.0f), 
                        float3{0.0f, 0.0f, 0.0f},
                        objectType
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassTorus(
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const float random2 = Hash11f(static_cast<uint32_t>(random * INT_MAX));
                    const float random3 = Hash11f(static_cast<uint32_t>(random2 * INT_MAX));
                    const float random4 = Hash11f(static_cast<uint32_t>(random3 * INT_MAX));
                    const float random5 = Hash11f(static_cast<uint32_t>(random4 * INT_MAX));
                    const float random6 = Hash11f(static_cast<uint32_t>(random5 * INT_MAX));
                    const float3 color = color::HSV2RGB(random6, 1.0f, 1.0f);
                    const auto currentColor = lerp(color, color::White(), normalized.x * 0.7f);
                    Scene.emplace_back(
                        location, float3{ 0.05f,0.05f,0.05f } * scale * objectScale,
                        Material::CreateGlassPureColor(currentColor, 1.3f + 2.0f * normalized.y, 5.0f + normalized.z * 50.0f, 1.0f, 0.1f), 
                        float3{random2 * 360.0f, random3 * 360.0f, random4 * 360.0f},
                        EObjectType::OBJ_SDF,
                        sdf::CreateTorus()
                    );
                }
            }
        }
        return Scene;
    }
    
    inline static std::vector<SceneObject> CreateGlassSDFVolume(
        const int volumeIndex,
        const float3 position = float3{0.0f, 0.0f, 0.0f},
        const float scale = 1.0f,
        const float objectScale = 1.0f,
        const int halfResolution = 2)
    {
        std::vector<SceneObject> Scene;
        for(int x = -halfResolution; x <= halfResolution; x++)
        {
            for(int y = -halfResolution; y <= halfResolution; y++)
            {
                for(int z = -halfResolution; z <= halfResolution; z++)
                {
                    const float3 normalized = {(static_cast<float>(x) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(y) + halfResolution) / halfResolution * 0.5f, (static_cast<float>(z) + halfResolution) / halfResolution * 0.5f};
                    const float3 location = float3{ x * 0.25f,y * 0.25f,z * 0.25f} * scale / static_cast<float>(halfResolution) + position;
                    const float random = Hash11f(static_cast<uint32_t>(normalized.x * 1024) | (static_cast<uint32_t>(normalized.y * 1024) << 10) | (static_cast<uint32_t>(normalized.z * 1024) << 20));
                    const float random2 = Hash11f(static_cast<uint32_t>(random * INT_MAX));
                    const float random3 = Hash11f(static_cast<uint32_t>(random2 * INT_MAX));
                    const float random4 = Hash11f(static_cast<uint32_t>(random3 * INT_MAX));
                    const float random5 = Hash11f(static_cast<uint32_t>(random4 * INT_MAX));
                    const float random6 = Hash11f(static_cast<uint32_t>(random5 * INT_MAX));
                    const float3 color = color::HSV2RGB(random6, 1.0f, 1.0f);
                    const auto currentColor = lerp(color, color::White(), normalized.z * 0.7f);
                    Scene.emplace_back(
                        location, float3{ 0.05f,0.05f,0.05f } * scale * objectScale,
                        Material::CreateGlassPureColor(currentColor, 1.6f + 2.0f * normalized.y, 5.0f + normalized.x * 25.0f, 1.0f, 0.1f), 
                        float3{random2 * 360.0f, random3 * 360.0f, random4 * 360.0f},
                        EObjectType::OBJ_SDF,
                        sdf::CreateSDFVolume(volumeIndex)
                    );
                }
            }
        }
        return Scene;
    }
};
