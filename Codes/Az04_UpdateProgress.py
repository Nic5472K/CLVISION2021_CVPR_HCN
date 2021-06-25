###===###
# Dependencies
import  random
import  math
import  numpy                   as      np
#---
import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F
#---
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

#---

def UPAz04(
        TaskID, My_TN,
        EoTE_Accuracy,
        Straight_After, Straight_Before,
        Old_Results, Best_Results
        ):
    print("=-="*5)
    p1 = "|| Target \t || Current \t"
    p2 = "|| Last \t || Best \t"
    p3 = "|| Diff2Last \t || Diff2Best"
    print(p1 + p2 + p3)
    print("---"*5)

    for atr in range(My_TN):
        CurrentID   = list(EoTE_Accuracy.keys())[atr]
        CurrentACC  = EoTE_Accuracy[CurrentID]

        #---
        if TaskID == atr:
            Straight_After.append(CurrentACC)
            print('>>>' * 5)

        #---
        if (TaskID + 1) == atr:
            Straight_Before.append(CurrentACC)

        #---
        PastAcc     = Old_Results[CurrentID]
        PrevBesAcc  = Best_Results[CurrentID]

        #---
        if CurrentACC > PrevBesAcc:
            Best_Results[CurrentID] = CurrentACC
            BestAcc = CurrentACC

        else:
            BestAcc = PrevBesAcc

        #---
        CurMinusLast = CurrentACC - PastAcc
        CurMinusBest = CurrentACC - BestAcc

        #---
        q1 =  "|| {} \t\t || {}% \t".format(
                                        CurrentID,
                                        round(CurrentACC * 100, 2)
                                       )
        q2 = "|| {}% \t || {}% \t".format(
                                        round(PastAcc    * 100, 2),
                                        round(BestAcc    * 100, 2)
                                       )
        q3a = '{}% '.format(round(CurMinusLast  * 100, 2))
        q3b = '{}% '.format(round(CurMinusBest  * 100, 2))

        #---
        if      (CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<+>'
        elif -1*(CurMinusLast  * 100) > 20:
                    q3a = q3a  + '<**>'
        elif -1*(CurMinusLast  * 100) > 10:
                    q3a = q3a  + '<*>'
        elif -1*(CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<.>'

        #---
        if      (CurMinusBest  * 100) >= 0:
            #---
            if atr > TaskID:
                q3b = q3b  + '<ZSL>'
            else:
                q3b = q3b  + '<+>'
                
        elif -1*(CurMinusBest  * 100) > 20:
                    q3b = q3b  + '<**>'
        elif -1*(CurMinusBest  * 100) > 10:
                    q3b = q3b  + '<*>'
        elif -1*(CurMinusBest  * 100) >  0:
                    q3b = q3b  + '<.>'        

        #---
        q3 = "|| {} \t || {}".format(q3a, q3b)

        #---
        print(q1 + q2 + q3)

    #---
    Old_Results = EoTE_Accuracy
        
    print("")

    return Straight_After, Straight_Before,\
           Old_Results, Best_Results
           

           









