import ast
import os

Mar20 = os.popen('dasgoclient --query="dataset dataset=/*/*Run2018D*20Mar2019*/*"').read()
Mar22 = os.popen('dasgoclient --query="dataset dataset=/*/*Run2018A*22Mar2019*/*"').read()

list1 = []
for run in ['Run2018A','Run2018D'][-2:] :
    for pd in ['BPH1','BPH2','BPH3','BPH4','BPH5'][:1] :
        for tier in ['/AOD','/MINIAOD','/RAW-RECO'][-2:] :
            for dataset in Mar20.split('\n') :
                if ( run in str(dataset) ) & ( pd in str(dataset) ) & ( tier in str(dataset) ) :
                    print 'dasgoclient --query="summary dataset=%s"'%dataset
                    string = os.popen('dasgoclient --query="summary dataset=%s"'%dataset).read()
                    dict1 = ast.literal_eval(string)
                    list1.append((run,pd,tier,list(dict1)[0]['nevents'],dataset))
            for dataset in Mar22.split('\n') :
                if ( run in str(dataset) ) & ( pd in str(dataset) ) & ( tier in str(dataset) ) :
                    print 'dasgoclient --query="summary dataset=%s"'%dataset
                    string = os.popen('dasgoclient --query="summary dataset=%s"'%dataset).read()
                    dict1 = ast.literal_eval(string)
                    list1.append((run,pd,tier,list(dict1)[0]['nevents'],dataset))
                    
for (run,pd,tier,nevents,dataset) in list1 :
    print "{:8s} {:4s} {:8} {:10.0f} {:s}".format(run,pd,tier.strip('/'),int(nevents),dataset)

print 'AOD     ',sum([ nevents for (run,pd,tier,nevents,dataset) in list1 if '/AOD' in tier ])
print 'MINIAOD ',sum([ nevents for (run,pd,tier,nevents,dataset) in list1 if '/MINIAOD' in tier ])
print 'RAW-RECO',sum([ nevents for (run,pd,tier,nevents,dataset) in list1 if '/RAW-RECO' in tier ])
