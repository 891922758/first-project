INC = -I inc/
OBJ_DIR=lib/
FUN_DIR=src/
TARGET = $(OBJ_DIR)libcutensordecomposition.a

FUN_CU=$(wildcard $(FUN_DIR)*.cu)
OBJ_O = $(addprefix $(OBJ_DIR),$(patsubst %.cu,%.o,$(notdir $(FUN_CU))))

$(TARGET):$(OBJ_O)
	ar cr $@ $^

$(OBJ_DIR)%.o:$(FUN_DIR)%.cu
	nvcc -c $< -o $@ -I inc/

.PHONY:clean
clean:
	rm $(OBJ_DIR)*
