from gp_gap import trial
for side,d,h in (('bot',0,0),('left',.6,.3),('bot',1.0,0)):
    for g in (0.012,0.014,0.016,0.018):
        print(side,d,g,trial(side,d,h,g),flush=True)
