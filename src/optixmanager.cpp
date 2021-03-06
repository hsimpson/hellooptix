#define NOMINMAX
#include "optixmanager.h"
#include "optixhelpers.h"
#include <iostream>
#include "program.h"
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <sstream>
#include <format>
#include <fstream>
#include "spdlog/spdlog.h"
#include "trackballController.h"

static void contextLogCallback(unsigned int level,
                               const char*  tag,
                               const char*  message,
                               void*) {
  const std::string fmt = "[{}]: {}";
  switch (level) {
    case 0:
      spdlog::debug(fmt, tag, message);
      break;
    case 1:
      spdlog::critical(fmt, tag, message);
      break;
    case 2:
      spdlog::error(fmt, tag, message);
      break;
    case 3:
      spdlog::warn(fmt, tag, message);
      break;
    case 4:
      spdlog::info(fmt, tag, message);
      break;

    default:
      break;
  }
}

OptixManager::OptixManager(std::shared_ptr<Scene> scene,
                           uint32_t               width,
                           uint32_t               height)
    : _scene(scene),
      _width(width),
      _height(height) {
  initOptix();
  createContext();
  createModule();
  createRayGenProgramGroup();
  createMissProgramGroup();
  createHitProgramGroup();
  buildAccel();
  createPipeline();
  createTextures();
  createShaderBindingTable();

  _outputBuffer = new CUDAOutputBuffer<uchar4>(CUDAOutputBufferType::CUDA_DEVICE, _width, _height);
  _accumBuffer  = new CUDAOutputBuffer<float4>(CUDAOutputBufferType::CUDA_DEVICE, _width, _height);

  _launchParams.frame.colorBuffer = _outputBuffer->map();
  _launchParams.frame.accumBuffer = _accumBuffer->map();
  _launchParams.frame.size        = make_uint2(_width, _height);
  _launchParams.frame.sampleIndex = 0;
}

OptixManager::~OptixManager() {
  delete _accumBuffer;
  delete _outputBuffer;

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_shaderBindingTable.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_shaderBindingTable.missRecordBase)));

  OPTIX_CHECK(optixPipelineDestroy(_pipeline));

  for (auto group : _hitProgramGroups) OPTIX_CHECK(optixProgramGroupDestroy(group));
  for (auto group : _missProgramGroups) OPTIX_CHECK(optixProgramGroupDestroy(group));
  for (auto group : _raygenProgramGroups) OPTIX_CHECK(optixProgramGroupDestroy(group));

  OPTIX_CHECK(optixModuleDestroy(_module));
  OPTIX_CHECK(optixDeviceContextDestroy(_optixContext));
}

void OptixManager::initOptix() {
  spdlog::info("Init optix...");
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  CUDA_CHECK(cudaFree(0));
  int numDevices;
  CUDA_CHECK(cudaGetDeviceCount(&numDevices));
  if (numDevices == 0)
    throw std::runtime_error("no CUDA capable devices found!");

  spdlog::info("Found {} CUDA devices", numDevices);

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());

  spdlog::info("optix successfully initialized");
}

void OptixManager::createContext() {
  spdlog::info("creating optix context ...");

  // for this sample, do everything on one device
  const int deviceID = 0;

  CUDA_CHECK(cudaSetDevice(deviceID));
  CUDA_CHECK(cudaStreamCreate(&_stream));

  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, deviceID));

  spdlog::info("running on device: {}", deviceProps.name);

  CUcontext cudaContext;
  CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
  if (cuRes != CUDA_SUCCESS) {
    spdlog::error("Error querying current context: error code {}", cuRes);
  }

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &_optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(_optixContext, contextLogCallback, nullptr, 4));
}

void OptixManager::createModule() {
  spdlog::info("setting up module ...");

  OptixModuleCompileOptions moduleCompileOptions = {};

  moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  _pipelineCompileOptions                = {};
  _pipelineCompileOptions.usesMotionBlur = false;
  // _pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  _pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  _pipelineCompileOptions.numPayloadValues      = 2;
  // _pipelineCompileOptions.numAttributeValues               = 2;
  _pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  _pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

  Program           prg("./src/assets/kernels/hellooptix.cu");
  const std::string ptxCode = prg.getPTX();

  char   log[2048];
  size_t sizeOfLog = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(_optixContext,
                                       &moduleCompileOptions,
                                       &_pipelineCompileOptions,
                                       ptxCode.c_str(),
                                       ptxCode.size(),
                                       log, &sizeOfLog,
                                       &_module));
  if (sizeOfLog > 1) PRINT(log);
}

void OptixManager::createRayGenProgramGroup() {
  spdlog::info("create raygen program group(s)");
  _raygenProgramGroups.resize(1);

  OptixProgramGroupOptions programGroupOptions = {};  // Initialize to zeros

  OptixProgramGroupDesc raygenProgGroupDesc    = {};
  raygenProgGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module            = _module;
  raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  char   log[2048];
  size_t sizeOfLog = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(_optixContext,
                                      &raygenProgGroupDesc,
                                      1,
                                      &programGroupOptions,
                                      log,
                                      &sizeOfLog,
                                      &_raygenProgramGroups[0]));
  if (sizeOfLog > 1) PRINT(log);
}

void OptixManager::createMissProgramGroup() {
  spdlog::info("create miss program group(s)");
  _missProgramGroups.resize(1);

  OptixProgramGroupOptions programGroupOptions = {};  // Initialize to zeros

  OptixProgramGroupDesc missProgGroupDesc  = {};
  missProgGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
  missProgGroupDesc.miss.module            = _module;
  missProgGroupDesc.miss.entryFunctionName = "__miss__radiance";

  char   log[2048];
  size_t sizeOfLog = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(_optixContext,
                                      &missProgGroupDesc,
                                      1,
                                      &programGroupOptions,
                                      log,
                                      &sizeOfLog,
                                      &_missProgramGroups[0]));
  if (sizeOfLog > 1) PRINT(log);
}

void OptixManager::createHitProgramGroup() {
  spdlog::info("create hit program group(s)");
  _hitProgramGroups.resize(1);

  OptixProgramGroupOptions programGroupOptions = {};  // Initialize to zeros

  OptixProgramGroupDesc hitProgGroupDesc        = {};
  hitProgGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitProgGroupDesc.hitgroup.moduleCH            = _module;
  hitProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  hitProgGroupDesc.hitgroup.moduleAH            = _module;
  hitProgGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  char   log[2048];
  size_t sizeOfLog = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(_optixContext,
                                      &hitProgGroupDesc,
                                      1,
                                      &programGroupOptions,
                                      log,
                                      &sizeOfLog,
                                      &_hitProgramGroups[0]));
  if (sizeOfLog > 1) PRINT(log);
}

void OptixManager::buildAccel() {
  spdlog::info("build acceleration structures ...");
  OptixTraversableHandle asHandle{0};

  size_t meshCount = _scene->meshes.size();

  _vertexBuffer.resize(meshCount);
  _normalBuffer.resize(meshCount);
  _texcoordBuffer.resize(meshCount);
  _indexBuffer.resize(meshCount);

  std::vector<OptixBuildInput> triangleInput(meshCount);
  std::vector<CUdeviceptr>     d_vertices(meshCount);
  std::vector<CUdeviceptr>     d_indices(meshCount);
  std::vector<uint32_t>        triangleInputFlags(meshCount);

  for (int meshID = 0; meshID < meshCount; meshID++) {
    // upload the model to the device: the builder
    auto& mesh = _scene->meshes[meshID];
    _vertexBuffer[meshID].alloc_and_upload(mesh.vertices);
    _indexBuffer[meshID].alloc_and_upload(mesh.indices);
    if (!mesh.normals.empty())
      _normalBuffer[meshID].alloc_and_upload(mesh.normals);
    if (!mesh.texcoords.empty())
      _texcoordBuffer[meshID].alloc_and_upload(mesh.texcoords);

    d_vertices[meshID] = _vertexBuffer[meshID].d_pointer();
    d_indices[meshID]  = _indexBuffer[meshID].d_pointer();

    triangleInput[meshID]                                   = {};
    triangleInput[meshID].type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertices.size();
    triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];

    triangleInput[meshID].triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
    triangleInput[meshID].triangleArray.numIndexTriplets   = (int)mesh.indices.size();
    triangleInput[meshID].triangleArray.indexBuffer        = d_indices[meshID];

    triangleInputFlags[meshID] = 0;

    triangleInput[meshID].triangleArray.flags                       = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords               = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
  }

  // ==================================================================
  // BLAS (Basic Linear Algebra Subprograms) setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.motionOptions.numKeys  = 1;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(_optixContext,
                                           &accelOptions,
                                           triangleInput.data(),
                                           (int)meshCount,  // num_build_inputs
                                           &blasBufferSizes));

  // ==================================================================
  // prepare compaction
  // ==================================================================

  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));

  OptixAccelEmitDesc emitDesc;
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();

  // ==================================================================
  // execute build (main stage)
  // ==================================================================

  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(_optixContext,
                              /* stream */ 0,
                              &accelOptions,
                              triangleInput.data(),
                              (int)meshCount,
                              tempBuffer.d_pointer(),
                              tempBuffer.sizeInBytes,

                              outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes,

                              &asHandle,

                              &emitDesc, 1));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // perform compaction
  // ==================================================================
  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize, 1);

  _asBuffer.alloc(compactedSize);
  OPTIX_CHECK(optixAccelCompact(_optixContext,
                                /*stream:*/ 0,
                                asHandle,
                                _asBuffer.d_pointer(),
                                _asBuffer.sizeInBytes,
                                &asHandle));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================
  outputBuffer.free();  // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();

  _launchParams.traversable = asHandle;
}

void OptixManager::createPipeline() {
  spdlog::info("create pipeline ...");

  const uint32_t                 maxTraceDepth = 2;
  std::vector<OptixProgramGroup> programGroups;
  for (auto group : _raygenProgramGroups) programGroups.push_back(group);
  for (auto group : _missProgramGroups) programGroups.push_back(group);
  for (auto group : _hitProgramGroups) programGroups.push_back(group);

  OptixPipelineLinkOptions pipelineLinkOptions = {};
  pipelineLinkOptions.maxTraceDepth            = maxTraceDepth;
  pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

  char   log[2048];
  size_t sizeOfLog = sizeof(log);

  OPTIX_CHECK(optixPipelineCreate(_optixContext,
                                  &_pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups.data(),
                                  (int)programGroups.size(),
                                  log,
                                  &sizeOfLog,
                                  &_pipeline));
  if (sizeOfLog > 1) PRINT(log);

  OptixStackSizes stackSizes = {};
  for (auto& programGroup : programGroups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, &stackSizes));
  }

  uint32_t directCallableStackSizeFromTraversal;
  uint32_t directCallableStackSizeFromState;
  uint32_t continuationStackSize;
  OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                         maxTraceDepth,
                                         0,  // maxCCDepth
                                         0,  // maxDCDEpth
                                         &directCallableStackSizeFromTraversal,
                                         &directCallableStackSizeFromState,
                                         &continuationStackSize));

  OPTIX_CHECK(optixPipelineSetStackSize(_pipeline,
                                        directCallableStackSizeFromTraversal,
                                        directCallableStackSizeFromState,
                                        continuationStackSize,
                                        1  // maxTraversableDepth
                                        ));
}

void OptixManager::createTextures() {
  spdlog::info("create textures ...");

  int numTextures = _scene->textures.size();
  _textureArrays.resize(numTextures);
  _textureObjects.resize(numTextures);

  for (int textureID = 0; textureID < numTextures; textureID++) {
    const auto& texture = _scene->textures[textureID];

    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t               width         = texture.resolution.x;
    int32_t               height        = texture.resolution.y;
    int32_t               numComponents = 4;
    int32_t               pitch         = width * numComponents * sizeof(uint8_t);
    channel_desc                        = cudaCreateChannelDesc<uchar4>();

    cudaArray_t& pixelArray = _textureArrays[textureID];
    CUDA_CHECK(cudaMallocArray(&pixelArray,
                               &channel_desc,
                               width, height));

    CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
                                   /* offset */ 0, 0,
                                   texture.pixelData.data(),
                                   pitch, pitch, height,
                                   cudaMemcpyHostToDevice));

    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = pixelArray;

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = cudaAddressModeWrap;
    tex_desc.addressMode[1]      = cudaAddressModeWrap;
    tex_desc.filterMode          = cudaFilterModeLinear;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    _textureObjects[textureID] = cuda_tex;
  }
}

void OptixManager::createShaderBindingTable() {
  spdlog::info("create shader binding table ...");
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (auto group : _raygenProgramGroups) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(group, &rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);
  }
  _raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  _shaderBindingTable.raygenRecord = _raygenRecordsBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (auto group : _missProgramGroups) {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(group, &rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  _missRecordsBuffer.alloc_and_upload(missRecords);
  _shaderBindingTable.missRecordBase          = _missRecordsBuffer.d_pointer();
  _shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
  _shaderBindingTable.missRecordCount         = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  int                         numObjects = (int)_scene->meshes.size();
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int meshID = 0; meshID < numObjects; meshID++) {
    HitgroupRecord rec;
    // all meshes use the same code, so all same hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(_hitProgramGroups[0], &rec));
    auto& mesh        = _scene->meshes[meshID];
    auto  color       = mesh.color;
    rec.data.color    = make_float3(color.r, color.g, color.b);
    rec.data.vertex   = (float3*)_vertexBuffer[meshID].d_pointer();
    rec.data.normal   = (float3*)_normalBuffer[meshID].d_pointer();
    rec.data.texcoord = (float2*)_texcoordBuffer[meshID].d_pointer();
    rec.data.index    = (uint3*)_indexBuffer[meshID].d_pointer();

    if (mesh.textureID >= 0) {
      rec.data.hasTexture = true;
      rec.data.texture    = _textureObjects[mesh.textureID];
    } else {
      rec.data.hasTexture = false;
    }
    hitgroupRecords.push_back(rec);
  }
  _hitRecordsBuffer.alloc_and_upload(hitgroupRecords);
  _shaderBindingTable.hitgroupRecordBase          = _hitRecordsBuffer.d_pointer();
  _shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  _shaderBindingTable.hitgroupRecordCount         = (int)hitgroupRecords.size();
}

void OptixManager::launch() {
  // std::cout << "sample index: " << _launchParams.frame.sampleIndex << std::endl;

  updateCamera();

  CUdeviceptr dParam;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParam), sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(dParam),
      &_launchParams,
      sizeof(_launchParams),
      cudaMemcpyHostToDevice));

  // std::cout << "sizeof(_launchParams): " << sizeof(_launchParams) << std::endl;
  // std::cout << "sizeof(Params): " << sizeof(Params) << std::endl;

  OPTIX_CHECK(optixLaunch(
      _pipeline,
      _stream,
      dParam,
      sizeof(Params),
      &_shaderBindingTable,
      _width,
      _height,
      /*depth=*/1));

  CUDA_SYNC_CHECK();

  _launchParams.frame.sampleIndex++;
}

void OptixManager::resize(uint32_t width, uint32_t height) {
  // minimized
  if (width == 0 || height == 0) return;

  // noop on no change
  if (width == _width && height == _height) return;

  _launchParams.frame.sampleIndex = 0;

  _width  = width;
  _height = height;

  _outputBuffer->unmap();
  _accumBuffer->unmap();
  _accumBuffer->resize(_width, _height);
  _outputBuffer->resize(_width, _height);
  _launchParams.frame.colorBuffer = _outputBuffer->map();
  _launchParams.frame.accumBuffer = _accumBuffer->map();

  _launchParams.frame.size = make_uint2(_width, _height);
}

void OptixManager::updateCamera() {
  if (!_scene->camera->needsUpdate())
    return;

  _scene->camera->setNeedsUpdate(false);

  _launchParams.frame.sampleIndex = 0;
  _launchParams.camera.eye        = vec3ToFloat3(_scene->camera->eye());
  glm::vec3 U, V, W;
  _scene->camera->getUVW(U, V, W);
  _launchParams.camera.U = vec3ToFloat3(U);
  _launchParams.camera.V = vec3ToFloat3(V);
  _launchParams.camera.W = vec3ToFloat3(W);
}

void OptixManager::writeImage(const std::string& imagePath) {
  // write the image as PPM
  auto imageData = _outputBuffer->getHostPointer();

  std::ofstream ofStream(imagePath, std::ios::out);
  if (ofStream.good()) {
    // header
    ofStream << "P3" << std::endl;
    ofStream << _width << " " << _height << std::endl;
    ofStream << "255" << std::endl;

    // pixel data
    std::stringstream ss;
    // std::string       s;
    for (int32_t i = _height - 1; i >= 0; --i) {
      for (int32_t j = 0; j < _width; j++) {
        int32_t idx = i * _width + j;

        ss << std::to_string(imageData[idx].x) << " "
           << std::to_string(imageData[idx].y) << " "
           << std::to_string(imageData[idx].z) << std::endl;
      }
    }
    ofStream << ss.rdbuf();
    ofStream.close();
  }
}
