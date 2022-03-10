add_llvm_component_library(LLVMDynamicVariant
  DynamicVariant.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/DynamicVariant

  DEPENDS
  intrinsics_gen

  COMPONENT_NAME
  dynamicvariant

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  )
