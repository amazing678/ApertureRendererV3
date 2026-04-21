#include "scene.hpp"
#include "camera.hpp"
#include "helper/scene_helper.h"

#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <tensor/tensor.cuh>
#include "spectrum//sample_ciexyz.cuh"
#include "spectrum/spectrum_lut.cuh"

std::vector<SceneObject> CreateScene()
{
    std::vector<SceneObject> Scene;
    /*
    std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f},
        Color::Red(),
        Color::Green(),
        Color::YellowLighten(0.9f) * 250.0f, 0.1f, {0.4f, -0.75f, 0.4f});
    Scene.insert(Scene.end(), CornellBox1.begin(), CornellBox1.end());
    Scene.emplace_back(
        float3{ 0.0f,-0.1f,0.0f }, float3{ 0.1f,0.4f,0.5f },
        Material::CreateGGXPureColor(Color::WhiteDarken(0.25f), 0.05f, 0.0f, 0.5f), // Color::WhiteDarken(0.25f)
        float3{0.0f, 0.0f, 0.0f},
        EObjectType::OBJ_CUBE
    );
    */
    //std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f}, Color::Red(), Color::Green(), Color::YellowLighten(0.9f) * 50.0f, 0.5f); // 
    std::vector<SceneObject> CornellBox1 = SceneHelper::CreateCornellBox({0.0f, 0.0f,0.0f}, color::White(), color::White(), color::YellowLighten(0.98f) * 50.0f, 0.5f); // 
    //std::vector<SceneObject> CornellBox1 = SceneHelper::CreateDiffuseCornellBox({0.0f, 0.0f,0.0f}, Color::Red(), Color::Green(), Color::YellowLighten(0.9f) * 50.0f, 0.5f); // 
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTestScene({0.0f, 0.0f,0.0f}, 1.0f, 20.0f, 0.5f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateSpectrumTestScene({0.0f, 0.0f,0.0f});
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateBigGlassTestScene({0.0f, 0.0f,0.0f}, 1.0);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateCubeTestScene({0.0f, 0.0f,0.0f}, 3.0f, 5.0f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateAbbeTestScene({0.0f, 0.0f,0.0f}, 5.0f, 2.0f);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTestScene2({0.0f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Green(), EObjectType::OBJ_CUBE);
    //std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassTorus({0.0f, 0.0f,0.0f}, 1.25f, 1.0f, 2);
    std::vector<SceneObject> GlassScene = SceneHelper::CreateGlassSDFVolume(0, {0.0f, 0.0f,0.0f}, 1.1f, 2.0f, 1);
    Scene.insert(Scene.end(), CornellBox1.begin(), CornellBox1.end());
    Scene.insert(Scene.end(), GlassScene.begin(), GlassScene.end());
    //
    /*
    std::vector<SceneObject> CornellBox2 = SceneHelper::CreateCornellBox({1.02f, 0.0f,0.0f},
        Color::Pink(),
        Color::Cyan(),
        Color::YellowLighten(0.9f) * 50.0f);
    std::vector<SceneObject> GlassMatrix = SceneHelper::CreateGlassTestScene2({1.02f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Red(), EObjectType::OBJ_SPHERE);
    Scene.insert(Scene.end(), CornellBox2.begin(), CornellBox2.end());
    Scene.insert(Scene.end(), GlassMatrix.begin(), GlassMatrix.end());
    //
    std::vector<SceneObject> CornellBox3 = SceneHelper::CreateCornellBox({-1.02f, 0.0f,0.0f},
        Color::Pink(),
        Color::Cyan(),
        Color::YellowLighten(0.9f) * 50.0f);
    std::vector<SceneObject> GlassMatrix2 = SceneHelper::CreateGlassTestScene2({-1.02f, 0.0f,0.0f}, 1.25f, 1.0f, 2, Color::Green(), EObjectType::OBJ_CUBE);
    Scene.insert(Scene.end(), CornellBox3.begin(), CornellBox3.end());
    Scene.insert(Scene.end(), GlassMatrix2.begin(), GlassMatrix2.end());
    //
    std::vector<SceneObject> CornellBox4 = SceneHelper::CreateCornellBox({0.0f, 1.04f,0.0f},
        Color::Purple(),
        Color::Yellow(),
        Color::YellowLighten(0.9f) * 150.0f,
        0.25);
    std::vector<SceneObject> GlassMatrix3 = SceneHelper::CreateGlassTestScene2({0.0f, 1.04f,0.0f}, 1.25f, 1.5f, 2, Color::Cyan(), EObjectType::OBJ_SPHERE, 0.25f);
    Scene.insert(Scene.end(), CornellBox4.begin(), CornellBox4.end());
    Scene.insert(Scene.end(), GlassMatrix3.begin(), GlassMatrix3.end());
    */
    return Scene;
}

int main()
{
    //spectrum::query::SelfTestSimple();
    //spectrum::sample::ValidationSampleCIEXYZ();
    //spectrum::ValidationSpectrum();
    
    srand(static_cast<unsigned>(time(NULL)));

    const std::vector<SceneObject> scene = CreateScene();
    SceneRenderer sceneRenderer(scene);

    {
        const AdditionalObjectInfo diamondSDF = sdf::CreateDiamond();
        sceneRenderer.AddVolumeToCacheManager(
            {64, 64, 64},
            {0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f},
            sdf::SDFDiamondFunctor{},
            diamondSDF.sdfInfo_
        );
    }

    const float3 cameraPosition = normalize(float3{ 0.0f,0.0f,0.8f }) * 1.1f;
     
    std::cout << "fill volume done." << std::endl;

    Camera camera(sceneRenderer, "test camera", 1280 / 2, 720 / 2);
    camera.SetPosition(cameraPosition);

    std::cout << "prepare rendering." << std::endl;
    RunGUI(camera, sceneRenderer);
    
    return 0;
}