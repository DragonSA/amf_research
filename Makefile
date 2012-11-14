DIRS=		presentation proposal report

.PHONY: 	all clean test ${DIRS}

all: ${DIRS}

clean:
	@${MAKE} -C common clean
.for dir in ${DIRS}
	@${MAKE} -C ${dir} -f ../Makefile.incl clean NAME=${dir}
.endfor

test:
	${PYTHON} -m unittest discover src/

.for dir in ${DIRS}
${dir}:
	@${MAKE} -C ${dir} -f ../Makefile.incl NAME=${dir}
.endfor

.include "commands.mk"
