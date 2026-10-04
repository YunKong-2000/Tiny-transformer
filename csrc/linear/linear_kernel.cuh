#pragma once

#include "linear.h"
#include "cutlass/gemm/device/gemm.h"

using TrainingForwardCTAShape = cutlass::gemm::GemmShape<128, 128, 8>;
using TrainingForwardWarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using TrainingForwardInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using BackwardCTAShape = cutlass::gemm::GemmShape<64, 64, 8>;
using BackwardWarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using BackwardInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using InferenceCTAShape = cutlass::gemm::GemmShape<8, 32, 8>;
using InferenceWarpShape = cutlass::gemm::GemmShape<4, 16, 8>;
using InferenceInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using OpClass = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm80;
using Operator = cutlass::arch::OpMultiplyAdd;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination< float, 1, float, float>;
using SwizzleOp = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr int kAlignmentA = 1;
constexpr int kAlignmentB = 1;
constexpr int kStages = 2;

using ForwardGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    TrainingForwardCTAShape,
    TrainingForwardWarpShape,
    TrainingForwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentA,
    kAlignmentB,
    false,
    Operator
    >;

using BackwardGemmX = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    BackwardCTAShape,
    BackwardWarpShape,
    BackwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentB,
    kAlignmentA,
    false,
    Operator
    >;

using BackwardGemmW = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    BackwardCTAShape,
    BackwardWarpShape,
    BackwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentB,
    kAlignmentA,
    false,
    Operator
    >;

using InferenceGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    InferenceCTAShape,
    InferenceWarpShape,
    InferenceInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentA,
    kAlignmentB,
    false,
    Operator
    >;
