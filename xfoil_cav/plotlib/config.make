
#=======================================#
# Makefile options for Xplot11 library  #
#   Set up or select a set of compile   #
#   options for your system             # 
#=======================================#


# Set library name 
PLTLIB = libPlt_gDP.a

# Some fortrans need trailing underscores in C interface symbols (see Xwin.c)
# This should work for most of the "unix" fortran compilers
CPPFLAGS += -DUNDERSCORE -I/usr/X11/include

FC = gfortran
CC  = gcc
DP = -fdefault-real-8

FFLAGS  += -fbounds-check -finit-real=inf -ffpe-trap=invalid,zero $(DP)
AR = ar r
RANLIB = ranlib 
LINKLIB = -lX11 
