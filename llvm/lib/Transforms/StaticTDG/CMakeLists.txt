add_llvm_component_library(LLVMStaticTDG
  StaticTDGIdent.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/StaticTDG

  DEPENDS
  intrinsics_gen

  COMPONENT_NAME
  statictdg

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  )
