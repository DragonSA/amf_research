# Makefile for figures
#
# FIGURES	List of figures that need to be built

SRC=		../src/

all: ${FIGURES}

clean:
	@echo "[common] clean"
	@rm -f fig*.pdf

.PHONY:		all clean
.SUFFIXES:	.py .pdf
.PATH.py:	${SRC}

.include "${SRC}Makefile.incl"
