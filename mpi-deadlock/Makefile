CC=mpicc
CFLAGS=-Wall -O3
SRCS=mixed-collective-barrier.c mixed-collective-deadlock.c opposite-order-barrier.c opposite-order-deadlock.c
OBJS=$(SRCS:.c=.o)
EXECS=$(SRCS:.c=)

.PHONY: all clean depend

all: $(EXECS)

$(EXECS): % : %.o
	$(CC) $(CFLAGS) -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(EXECS) *~

depend: $(SRCS)
	makedepend $(INCLUDES) $^

