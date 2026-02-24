#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Handle various system requirements for trailing underscores, or other 
   fortran-to-C interface shenanigans thru defines for routine names  
   The provided set gives the option of setting a compile flag -DUNDERSCORE
   to include underscores on C routine name symbols */

#ifdef UNDERSCORE

#define GETOSFILE  getosfile_ 

#endif

#define DEFAULT_OSFILE "/usr/lib/xfoil/osmap.dat"

void
GETOSFILE(osfile,len)
     char *osfile;
     int  *len;
{  
   char *bufp;
   int l;

/* get environment variable OSMAP for location of OS map data file */
   bufp = getenv("OSMAP");

   /* printf("bufp: %s\n",bufp);
      printf("osfile: %s\n",osfile); */

   if(bufp){ 
      strncpy(osfile,bufp, 128);
      l = strlen(bufp);
      }
    else {
      strncpy(osfile,DEFAULT_OSFILE, 128);
      l = strlen(DEFAULT_OSFILE);
      }

   *len = l;

   /*   printf("len %d\n",*len); */
}

