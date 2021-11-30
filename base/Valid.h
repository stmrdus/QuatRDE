#ifndef VALID_H
#define VALID_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

INT lastValidHead = 0;
INT lastValidTail = 0;
	
REAL l_valid_filter_tot = 0;
REAL r_valid_filter_tot = 0;

REAL l3_valid_filter_tot = 0;
REAL r3_valid_filter_tot = 0;

REAL l1_valid_filter_tot = 0;
REAL r1_valid_filter_tot = 0;

REAL r_valid_filter_rank = 0;
REAL l_valid_filter_rank = 0;

REAL r_valid_filter_reci_rank = 0;
REAL l_valid_filter_reci_rank = 0;

extern "C"
void validInit() {
    lastValidHead = 0;
    lastValidTail = 0;
    l_valid_filter_tot = 0;
    r_valid_filter_tot = 0;

    l3_valid_filter_tot = 0;
    r3_valid_filter_tot = 0;

    l1_valid_filter_tot = 0;
    r1_valid_filter_tot = 0;

    r_filter_rank = 0;
    l_filter_rank = 0;

    r_filter_reci_rank = 0;
    l_filter_reci_rank = 0;
}

extern "C"
void getValidHeadBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
	ph[i] = i;
	pt[i] = validList[lastValidHead].t;
	pr[i] = validList[lastValidHead].r;
    }
}

extern "C"
void getValidTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
	ph[i] = validList[lastValidTail].h;
	pt[i] = i;
	pr[i] = validList[lastValidTail].r;
    }
}

extern "C"
void validHead(REAL *con) {
    INT h = validList[lastValidHead].h;
    INT t = validList[lastValidHead].t;
    INT r = validList[lastValidHead].r;
    REAL minimal = con[h];
    INT l_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++) {
	    if (j != h) {
	        REAL value = con[j];
   	        if (value < minimal && ! _find(j, t, r)) {
		        l_filter_s += 1;
	        }
	    }
    }
    if (l_filter_s < 10) l_valid_filter_tot += 1;
    if (l_filter_s < 3) l3_valid_filter_tot += 1;
    if (l_filter_s < 1) l1_valid_filter_tot += 1;
    l_valid_filter_rank += (l_filter_s+1);
    l_valid_filter_reci_rank += 1.0/(l_filter_s+1);
    lastValidHead ++;
  //  printf("head: l_valid_filter_tot = %f | l_filter_hit10 = %f\n", l_valid_filter_tot, l_valid_filter_tot / lastValidHead);
}

extern "C"
void validTail(REAL *con) {
    INT h = validList[lastValidTail].h;
    INT t = validList[lastValidTail].t;
    INT r = validList[lastValidTail].r;
    REAL minimal = con[t];
    INT r_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++) {
	    if (j != t) {
	        REAL value = con[j];
	        if (value < minimal && ! _find(h, j, r)) {
	            r_filter_s += 1;
	        }
	    }
    }
    if (r_filter_s < 10) r_valid_filter_tot += 1;
    if (r_filter_s < 3) r3_valid_filter_tot += 1;
    if (r_filter_s < 1) r1_valid_filter_tot += 1;
    r_valid_filter_rank += (1+r_filter_s);
    r_valid_filter_reci_rank += 1.0/(1+r_filter_s);
    lastValidTail ++;
//    printf("tail: r_valid_filter_tot = %f | r_filter_hit10 = %f\n", r_valid_filter_tot, r_valid_filter_tot / lastValidTail);
}

extern "C"
void getResult(REAL res[]){
    l_valid_filter_tot /= validTotal;
    r_valid_filter_tot /= validTotal;
    validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2;

    l3_valid_filter_tot /= validTotal;
    r3_valid_filter_tot /= validTotal;
    validHit3 = (l3_valid_filter_tot + r3_valid_filter_tot) / 2;

    l1_valid_filter_tot /= validTotal;
    r1_valid_filter_tot /= validTotal;
    validHit1 = (l1_valid_filter_tot + r1_valid_filter_tot) / 2;

    l_valid_filter_rank /= validTotal;
    r_valid_filter_rank /= validTotal;
    validMeanRank = (l_valid_filter_rank + r_valid_filter_rank) / 2;

    l_valid_filter_reci_rank /= validTotal;
    r_valid_filter_reci_rank /= validTotal;
    validMeanReciprocalRank = (l_valid_filter_reci_rank+ r_valid_filter_reci_rank) / 2;

    res[0] = validMeanReciprocalRank;
    res[1] = validMeanRank;
    res[2] = validHit10;
    res[3] = validHit3;
    res[4] = validHit1;
}

REAL validHit10 = 0;
extern "C"
REAL getValidHit10() {
    l_valid_filter_tot /= validTotal;
    r_valid_filter_tot /= validTotal;
    validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2;
    printf("result hits@10: %f\n", validHit10);
    return validHit10;
}

REAL validHit3 = 0;
extern "C"
REAL getValidHit3() {
    l3_valid_filter_tot /= validTotal;
    r3_valid_filter_tot /= validTotal;
    validHit3 = (l3_valid_filter_tot + r3_valid_filter_tot) / 2;
    printf("result hits@3: %f\n", validHit3);
    return validHit3;
}

REAL validHit1 = 0;
extern "C"
REAL getValidHit1() {
    l1_valid_filter_tot /= validTotal;
    r1_valid_filter_tot /= validTotal;
    validHit1 = (l1_valid_filter_tot + r1_valid_filter_tot) / 2;
    printf("result hits@1: %f\n", validHit1);
    return validHit1;
}

REAL validMeanRank = 0;
extern "C"
REAL getValidMeanRank(){
    l_valid_filter_rank /= validTotal;
    r_valid_filter_rank /= validTotal;
    validMeanRank = (l_valid_filter_rank + r_valid_filter_rank) / 2;
    printf("result mean rank: %f\n", validMeanRank);
    return validMeanRank;
}

REAL validMeanReciprocalRank= 0;
extern "C"
REAL getValidMeanReciprocalRank(){
    l_valid_filter_reci_rank /= validTotal;
    r_valid_filter_reci_rank /= validTotal;
    validMeanReciprocalRank = (l_valid_filter_reci_rank+ r_valid_filter_reci_rank) / 2;
    printf("result mean rank: %f\n", validMeanReciprocalRank);
    return validMeanReciprocalRank;
}
#endif