from __future__ import division
import os
import re
import math

hsNdcgRelScore = {'2':3,'1':1,'0':0}
hsPrecisionRel={'2':1,'1':1,'0':0}
iMaxPosition = 10


def FoldPrecAtN(hsResult):
	qids=sorted(hsResult.items(),key=lambda d:d[0])
	prec=[0]*iMaxPosition
	for qid in qids:
		pN = hsResult[qid[0]]['PatN']
		#map_q =hsResult[qid[0]]['MAP']
		for iPos in range(len(pN)):
			prec[iPos] += pN[iPos]
	for iPos in range(iMaxPosition):
		#prec [iPos] =round(prec [iPos]/len(qids),6)
		prec [iPos] =prec [iPos]/len(qids)
	return prec

def FoldMap(hsResult):
	map=0
	qids=sorted(hsResult.items(),key=lambda d:d[0])
	for qid in qids:
		map_q =hsResult[qid[0]]['MAP']
		map += map_q
	map =round(map/len(qids),6)
	return map


def FoldNdcg(hsResult):
	"""path_adaRank='G:\\benchmark\\learning_to_rank\\letor40\\baselines\\AdaRank\\'
	FOUT=open(os.path.join(path_adaRank,'data\\output334'),'w')"""
	qids=sorted(hsResult.items(),key=lambda d:d[0])
	ndcg=[0]*iMaxPosition
	for qid in qids:
		ndcg_q = hsResult[qid[0]]['NDCG']
		
		for iPos in range(iMaxPosition):
			if iPos< len(ndcg_q):
				ndcg[iPos] =ndcg[iPos]+ndcg_q[iPos]
		"""FOUT.write(str(qid[0])+'\t')
		FOUT.write(str(ndcg_q[1]))
		FOUT.write('\n')"""
	for iPos in range(len(ndcg)):
		ndcg[iPos] =round(ndcg [iPos]/len(qids),6)
	"""FOUT.write(str(ndcg[0]))
	FOUT.write('\n')"""
	return ndcg
		
def FoldMeanNdcg(hsResult):
	qids=sorted(hsResult.items(),key=lambda d:d[0])
	meanNdcg = 0.0
	for qid in qids:
		meanNdcg_q = hsResult[qid[0]]['MeanNDCG']
		meanNdcg += meanNdcg_q
	meanNdcg =round(meanNdcg/len(qids),6)
	return meanNdcg

"""def OuputResults(fnOut,hsResult):
	try:
		FOUT=open(fnOut,'a')
	finally:
		if FOUT:
			print 'Invalid command line.\n'
			print 'Open \$fnOut\' failed.\n'
			FOUT.close()
			exit -2

    qids=sorted(hsResult.items(),lambda d:d[0])
    
#Precision@N and MAP
# modified by Jun Xu, March 3, 2009
# changing the output format
    FOUT.write('qid\tP\@1\tP\@2\tP\@3\tP\@4\tP\@5\tP\@6\tP\@7\tP\@8\tP\@9\tP\@10\tMAP\n')
#---------------------------------------------
    prec=[0]*iMaxPosition
    map = 0
	for qid in qids.items():
# modified by Jun Xu, March 3, 2009
# output the real query id    
	
        pN = hsResult[qid[0]]['PatN']
        map_q =hsResult[qid[0]]['MAP']
        if flag == 1:
            FOUT.write('qid\t')
            for iPos in range(len(iMaxPosition)):
				FOUT.write('%.4f\t', pN[iPos])
            FOUT.write('%.4f\t', map_q)
			
        for iPos in range(len(iMaxPosition)):
			prec[iPos] += pN[iPos]
        map += map_q
		
    print FOUT "Average\t"
    for iPos in range(len(iMaxPosition)):
        prec[iPos] /= len(qids)
        FOUT.write('%.4f\t', prec[iPos])
		
    map /= len(qids)
    FOUT.write('%.4f\t', map)
    
#NDCG and MeanNDCG
# modified by Jun Xu, March 3, 2009
# changing the output format
    FOUT.write('qid\tNDCG\@1\tNDCG\@2\tNDCG\@3\tNDCG\@4\tNDCG\@5\tNDCG\@6\tNDCG\@7\tNDCG\@8\tNDCG\@9\tNDCG\@10\tMeanNDCG\n')
#---------------------------------------------
    ndcg=[0]*iMaxPosition
    meanNdcg = 0;
    for qid in qids.items():
# modified by Jun Xu, March 3, 2009
# output the real query id
        
        ndcg_q = hsResult[qid[0]]['NDCG']
        meanNdcg_q = hsResult[qid[0]]['MeanNDCG']
        if flag == 1:
            FOUT.write('qid\t')
            for iPos in range(len(iMaxPosition)):
				FOUT.write('%.4f\t', ndcg_q[iPos])
            FOUT.write('%.4f\t', meanNdcg_q)
		
		for iPos in range(len(iMaxPosition)):
			ndcg[iPos] += ndcg_q[iPos]
        meanNdcg += meanNdcg_q
        
	FOUT.write('Average\t')
    for iPos in range(len(iMaxPosition)):
        ndcg[iPos] /= len(qids)
        FOUT.write('%.4f\t', ndcg[iPos])
		
    meanNdcg /= len(qids)
    FOUT.write('%.4f\t', meanNdcg)
	FOUT.close()
	return map
"""


def EvalQuery(pHash):
	hsResults={}
	
	qids = sorted(pHash.items(), key=lambda d:d[0]) 
	"""path_adaRank='G:\\benchmark\\learning_to_rank\\letor40\\baselines\\AdaRank\\'
	FOUT=open(os.path.join(path_adaRank,'data\\output222'),'w')
	FOUT2=open(os.path.join(path_adaRank,'data\\output334'),'w')"""
	for qid in qids:
		tmpDid =sorted(pHash[qid[0]].items(),key=lambda d:d[1]['lineNum']) 
		
		docids =sorted(tmpDid,key=lambda d:float(d[1]['pred']),reverse=True)
		"""if qid[0]=='16625':
			FOUT.write(qid[0])
			for i in range(len(docids)):
				FOUT.write(str(docids[i])+'\n')
			FOUT.write('\n')
			FOUT.close"""
		rates=[0]*len(docids)

		for iPos in range(len(rates)):
			rates[iPos] = pHash[qid[0]][docids[iPos][0]]['label']
			
		
		map  = MAP(rates)
		PAtN = PrecisionAtN(iMaxPosition, rates)
# modified by Jun Xu, calculate all possible positions' NDCG for MeanNDCG
        #my @Ndcg = NDCG($iMaxPosition, @rates)
        
		Ndcg = NDCG(len(rates), rates)
		meanNdcg = 0.0
		for iPos in range(len(Ndcg)):
			meanNdcg += Ndcg[iPos]
		meanNdcg /= len(Ndcg)
		tempResults={'PatN':PAtN,'MAP':map,'NDCG':Ndcg,'MeanNDCG':meanNdcg}
		hsResults[qid[0]]=tempResults
		"""if qid[0]=='16625':
			FOUT2.write( qid[0]+str(tempResults))
			FOUT2.write('\n')
	
	FOUT2.close"""
	return hsResults
	
def ReadInputFiles(fnFeature, fnPred):
	hsQueryDocLabelScore={}
	try:
		FIN_Feature=open(fnFeature,'r')
	finally:
		if not FIN_Feature:
			print 'Invalid command line.\n'
			print 'Open \$fnFeature\' failed.\n'
			FIN_Feature.close()
			exit -2

	try:
		FIN_Pred=open(fnPred,'r')
	finally:
		if not FIN_Pred:
			print 'Invalid command line.\n'
			print 'Open \$fnPred\' failed.\n'
			FIN_Pred.close()
			exit -2
	
	lineNum = 0
	lnFea=FIN_Feature.readline()
	while lnFea!='':
		
		lnFea=lnFea.strip('\n')
		lineNum +=1
		predScore=FIN_Pred.readline()
		if predScore is None:
			print 'Error to read $fnPred at line $lineNum.\n'
			exit -2
		predScore=predScore.strip('\n')
		m=re.match(r'^(\d+) qid\:([^\s]+).*?\#docid = ([^\s]+) inc = ([^\s]+) prob = ([^\s]+)$',lnFea)
		if m:
			label = m.group(1)
			qid = m.group(2)
			did = m.group(3)
			inc = m.group(4)
			prob= m.group(5)
			#print label
			temp_did={'label':label,'inc':inc,'prob':prob,'pred':predScore,'lineNum':lineNum}
			#print temp_did
			if hsQueryDocLabelScore.has_key(qid):
				hsQueryDocLabelScore[qid][did]=temp_did
				#hsQueryDocLabelScore[qid].update(add_to_qid)
			else:
				hsQueryDocLabelScore[qid]={}
				hsQueryDocLabelScore[qid][did]=temp_did
		else:
			print 'Error to parse fnFeature at line'+str(lineNum)+':\nlnFea\n'
			exit -2
		lnFea=FIN_Feature.readline()
 
	FIN_Feature.close()
	FIN_Pred.close()
	return hsQueryDocLabelScore

#each qid's MAP	
def MAP(rates):
	numRelevant = 0
	avgPrecision = 0.0
	for iPos in range(len(rates)):
		if  hsPrecisionRel[str(rates[iPos])] == 1:
			numRelevant+=1
			#avgPrecision += round((numRelevant / (iPos + 1)),6 )
			avgPrecision += (numRelevant / (iPos + 1))
	if  numRelevant == 0:
		return 0.0
	return avgPrecision / numRelevant

#Prec of each qid's the Nth doc	
def PrecisionAtN(topN,rates):
	numRelevant = 0
	PrecN=[0.0]*topN
#   modified by Jun Xu, 2009-4-24.
#   if # retrieved doc <  $topN, the P@N will consider the hole as irrelevant
#    for(my $iPos = 0  $iPos < $topN && $iPos < $#rates + 1 $iPos ++)
#
	for iPos in range(topN):
		if  iPos < len(rates):
			r = rates[iPos]
		else:
			r = 0
		if  hsPrecisionRel[str(r)] == 1:
			numRelevant+=1
		PrecN[iPos] = float('%.6f'%(numRelevant / (iPos + 1)))
		
	return PrecN

def f(x):
	return hsNdcgRelScore[str(x)]
	
def NDCG(topN, rates):
	ndcg=[0.0]*topN
	dcg = DCG(topN,rates)
	stRates=[0]*len(rates)
	
	stRates = sorted(rates,key=f,reverse=True)
	bestDcg = DCG(topN,stRates)
    
	iPos =0
	while iPos < topN and iPos < len(rates):
		if (bestDcg[iPos] != 0):
			ndcg[iPos] = float('%.6f'%(dcg[iPos] / bestDcg[iPos] ))
		iPos+=1
	return ndcg	
def DCG(topN,rates):
	dcg=[0.0]*topN
	dcg[0] = hsNdcgRelScore[str(rates[0])]
#   Modified by Jun Xu, 2009-4-24
#   if # retrieved doc <  $topN, the NDCG@N will consider the hole as irrelevant
#    for(my $iPos = 1 $iPos < $topN && $iPos < $#rates + 1 $iPos ++)
#
	for iPos in range(1,topN):
		r=0
		if (iPos < len(rates)):
			r = rates[iPos]
		else:
			r = 0
		if (iPos < 2):
			dcg[iPos] = dcg[iPos - 1] + hsNdcgRelScore[str(r)]
		else:
			dcg[iPos] = dcg[iPos - 1] + round(hsNdcgRelScore[str(r)] * math.log(2.0) / math.log(iPos + 1.0),6)
	return dcg
