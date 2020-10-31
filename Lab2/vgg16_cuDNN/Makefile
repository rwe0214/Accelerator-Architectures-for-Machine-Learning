NVCC = nvcc
LDFLAGS = -lcudnn_static -lculibos -lcublas_static -lcublasLt_static
ARCH = -arch=sm_70
EXE = vgg16
SRCDIR = .

SRCS = vgg16_cuDNN.cu

all: $(EXE)

$(EXE): $(SRCS)
	$(NVCC) -g $(ARCH) -o $@ $? $(LDFLAGS)

run:
	@./$(EXE)

dump:
	cuobjdump --dump-ptx $(EXE) > ptx_assembly.txt

format:
	clang-format -i $(SRCDIR)/*.cu

clean:
	@rm -rf $(EXE)
