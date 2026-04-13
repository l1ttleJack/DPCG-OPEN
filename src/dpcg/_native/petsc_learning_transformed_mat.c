#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>

typedef struct {
  Mat A;
  Mat L;
  Vec work1;
  Vec work2;
} DPCGLearningShellCtx;

typedef struct {
  Mat L;
  Vec work;
} DPCGLearningPCShellCtx;

static PetscErrorCode DPCGLearningShellMult(Mat shell, Vec x, Vec y)
{
  DPCGLearningShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &ctx));
  PetscCheck(ctx != NULL, PetscObjectComm((PetscObject)shell), PETSC_ERR_ARG_WRONGSTATE, "learning shell context is NULL");
  PetscCall(MatMultTranspose(ctx->L, x, ctx->work1));
  PetscCall(MatMult(ctx->A, ctx->work1, ctx->work2));
  PetscCall(MatMult(ctx->L, ctx->work2, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningShellCreateVecs(Mat shell, Vec *right, Vec *left)
{
  DPCGLearningShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &ctx));
  PetscCheck(ctx != NULL, PetscObjectComm((PetscObject)shell), PETSC_ERR_ARG_WRONGSTATE, "learning shell context is NULL");
  PetscCall(MatCreateVecs(ctx->A, right, left));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningShellDestroy(Mat shell)
{
  DPCGLearningShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &ctx));
  if (ctx != NULL) {
    PetscCall(VecDestroy(&ctx->work1));
    PetscCall(VecDestroy(&ctx->work2));
    PetscCall(MatDestroy(&ctx->A));
    PetscCall(MatDestroy(&ctx->L));
    PetscCall(PetscFree(ctx));
    PetscCall(MatShellSetContext(shell, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode dpcg_learning_shell_attach(void *shell_handle, void *a_handle, void *l_handle)
{
  Mat shell = (Mat)shell_handle;
  Mat A = (Mat)a_handle;
  Mat L = (Mat)l_handle;
  DPCGLearningShellCtx *ctx = NULL;
  PetscInt m = 0, n = 0, M = 0, N = 0;

  PetscFunctionBegin;
  PetscCheck(shell != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "shell mat handle is NULL");
  PetscCheck(A != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "A mat handle is NULL");
  PetscCheck(L != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "L mat handle is NULL");

  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatSetSizes(shell, m, n, M, N));
  PetscCall(MatSetType(shell, MATSHELL));
  PetscCall(MatSetVecType(shell, VECCUDA));

  PetscCall(PetscNew(&ctx));
  ctx->A = A;
  ctx->L = L;
  PetscCall(PetscObjectReference((PetscObject)ctx->A));
  PetscCall(PetscObjectReference((PetscObject)ctx->L));
  PetscCall(MatCreateVecs(ctx->A, &ctx->work2, &ctx->work1));

  PetscCall(MatShellSetContext(shell, ctx));
  PetscCall(MatShellSetOperation(shell, MATOP_MULT, (void (*)(void))DPCGLearningShellMult));
  PetscCall(MatShellSetOperation(shell, MATOP_CREATE_VECS, (void (*)(void))DPCGLearningShellCreateVecs));
  PetscCall(MatShellSetOperation(shell, MATOP_DESTROY, (void (*)(void))DPCGLearningShellDestroy));
  PetscCall(MatSetUp(shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningPCShellApply(PC pc, Vec x, Vec y)
{
  DPCGLearningPCShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx != NULL, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "learning PC shell context is NULL");
  PetscCall(MatMultTranspose(ctx->L, x, ctx->work));
  PetscCall(MatMult(ctx->L, ctx->work, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningPCShellApplySymmetricLeft(PC pc, Vec x, Vec y)
{
  DPCGLearningPCShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx != NULL, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "learning PC shell context is NULL");
  PetscCall(MatMult(ctx->L, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningPCShellApplySymmetricRight(PC pc, Vec x, Vec y)
{
  DPCGLearningPCShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx != NULL, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "learning PC shell context is NULL");
  PetscCall(MatMultTranspose(ctx->L, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningPCShellApplyTranspose(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(DPCGLearningPCShellApply(pc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DPCGLearningPCShellDestroy(PC pc)
{
  DPCGLearningPCShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  if (ctx != NULL) {
    PetscCall(VecDestroy(&ctx->work));
    PetscCall(MatDestroy(&ctx->L));
    PetscCall(PetscFree(ctx));
    PetscCall(PCShellSetContext(pc, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode dpcg_learning_pcshell_attach(void *pc_handle, void *l_handle)
{
  PC pc = (PC)pc_handle;
  Mat L = (Mat)l_handle;
  DPCGLearningPCShellCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscCheck(pc != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PC handle is NULL");
  PetscCheck(L != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "L mat handle is NULL");

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PetscNew(&ctx));
  ctx->L = L;
  PetscCall(PetscObjectReference((PetscObject)ctx->L));
  PetscCall(MatCreateVecs(ctx->L, &ctx->work, NULL));

  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetApply(pc, DPCGLearningPCShellApply));
  PetscCall(PCShellSetApplySymmetricLeft(pc, DPCGLearningPCShellApplySymmetricLeft));
  PetscCall(PCShellSetApplySymmetricRight(pc, DPCGLearningPCShellApplySymmetricRight));
  PetscCall(PCShellSetApplyTranspose(pc, DPCGLearningPCShellApplyTranspose));
  PetscCall(PCShellSetDestroy(pc, DPCGLearningPCShellDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}
