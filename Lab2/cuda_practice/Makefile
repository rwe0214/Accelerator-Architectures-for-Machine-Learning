NVCC = nvcc
EXES = MatAdd-pitch

all: $(EXES)

%: %.cu
	@$(NVCC) -o $@ $?

run:
	@./$(EXES)

clean:
	@rm -rf $(EXES)