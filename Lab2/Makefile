MATADD_PITCH = cuda_practice
VGG16_CUDNN = vgg16_cuDNN
EXE1 = MatAdd-pitch
EXE2 = vgg16

all:
	@echo 'Compiling $(EXE1)'
	@make -C $(MATADD_PITCH)
	@mv $(MATADD_PITCH)/$(EXE1) $(EXE1)
	@echo 'Compiling $(EXE2)'
	@make -C $(VGG16_CUDNN)
	@mv $(VGG16_CUDNN)/$(EXE2) $(EXE2)

run_matadd_pitch:
	./$(EXE1)

run_vgg16:
	./$(EXE2)

clean:
	@echo 'Deleting $(EXE1) $(EXE2)'
	@make -C $(MATADD_PITCH) clean
	@make -C $(VGG16_CUDNN) clean
	@rm -f $(EXE1) $(EXE2)
