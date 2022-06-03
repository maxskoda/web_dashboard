# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import urllib
from requests_html import HTMLSession
import re

from main import get_values



inst = "INTER"
lambda_min = 1.8
lambda_max = 15
trans_SM = 'TRANS_SM'
trans = 'TRANS'
dq_q = 0.03
TRANS_ROI = '70-90'
ROI = '70-90'

url='http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'
session = HTMLSession()
values = get_values(session)

# ISISJournalGetExperimentRuns(Cycle='21_2', InvestigationId='2210179', OutputWorkspace='RB2210179')
# runs = mtd['RB2210179']
#
# print(runs.column(1))

ReflectometryISISLoadAndProcess(InputRunList='65272', ThetaIn=0.8,
                                AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90',
                                WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4,
                                MonitorIntegrationWavelengthMax=10,
                                FirstTransmissionRunList='65274', SecondTransmissionRunList='65275',
                                StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False,
                                TransmissionProcessingInstructions='70-90',
                                MomentumTransferMin=0.010321317306126728,
                                MomentumTransferStep=0.055433662337842131,
                                MomentumTransferMax=0.1168874036214391,
                                OutputWorkspaceBinned='IvsQ_binned_65272',
                                OutputWorkspace='IvsQ_65272',
                                OutputWorkspaceTransmission='TRANS_SM')

ReflectometryISISLoadAndProcess(InputRunList='65273', ThetaIn=2.3, AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='67-95', WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, FirstTransmissionRunList='65276', SecondTransmissionRunList='65277', StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90', MomentumTransferMin=0.029666234509808882, MomentumTransferStep=0.055446760622640492, MomentumTransferMax=0.33612056568876092, OutputWorkspaceBinned='IvsQ_binned_65273', OutputWorkspace='IvsQ_65273', OutputWorkspaceTransmission='TRANS')
# Stitch1DMany(InputWorkspaces='IvsQ_65272,IvsQ_65273', OutputWorkspace='IvsQ_65272_65273', Params='-0.055434', OutScaleFactors='0.841361')


# script = """from CaChannel import CaChannel, CaChannelException, ca\n"""

script = """
values = get_values(session)\n
theta_in=float(values['THETA'])\n
qmin=4*3.1415/"""+str(lambda_max)+"""*(theta_in*3.141/180)
qmax=4*3.1415/"""+str(lambda_min)+"""*(theta_in*3.141/180)

if theta_in<1.0:\n
\t trans='"""+trans_SM+"""'\n
else:\n
\t trans='"""+trans+"""'\n

ReflectometryISISLoadAndProcess(InputRunList=input, ThetaIn=theta_in, SummationType='SumInQ', ReductionType='DivergentBeam', 
                        AnalysisMode='MultiDetectorAnalysis', 
                        ProcessingInstructions='"""+ROI+"""', 
                        WavelengthMin=1.5, WavelengthMax=17, 
                        I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, 
                        MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, SubtractBackground=True, 
                        BackgroundCalculationMethod='AveragePixelFit', 
                        FirstTransmissionRunList=trans, StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, 
                        TransmissionProcessingInstructions='"""+TRANS_ROI+"""', Debug=True, 
                        MomentumTransferMin=qmin, MomentumTransferStep="""+str(dq_q)+""", 
                        MomentumTransferMax=qmax, OutputWorkspaceBinned='0_IvsQ_binned', 
                        OutputWorkspace='0_IvsQ', OutputWorkspaceWavelength='0_IvsLam', 
                        OutputWorkspaceTransmission='TRANS_LAM_0', 
                        OutputWorkspaceFirstTransmission='TRANS_LAM_0a', OutputWorkspaceSecondTransmission='TRANS_LAM_0b')

output="0_IvsQ_binned" """

StartLiveData(Instrument=inst, ProcessingScript=script, AccumulationMethod='Replace',
              UpdateEvery=10, OutputWorkspace='0_IvsQ_binned')



fig = plt.figure()  # Create figure
axes = fig.add_subplot(111) # Add subplot (dont worry only one plot appears)

axes.set_autoscale_on(True) # enable autoscale
axes.autoscale_view(True,True,True)

xd = mtd['0_IvsQ_binned'].dataX(0)[:-1]
yd = mtd['0_IvsQ_binned'].dataY(0)
ed = mtd['0_IvsQ_binned'].dataE(0)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(xd, yd, ed)


while True:
    # Generate figure
    xd = mtd['0_IvsQ_binned'].dataX(0)[:-1]
    yd = mtd['0_IvsQ_binned'].dataY(0)
    ed = mtd['0_IvsQ_binned'].dataE(0)

    xarray = np.array(xd)
    yarray = np.array(yd)
    earray = np.array(ed)

    data = np.column_stack([xarray, yarray, earray])
    datafile_path = "text.csv"
    np.savetxt(datafile_path, data, fmt=['%.10e', '%.10e', '%.10e'])
    # line1.set_ydata(yd)
    fig.canvas.draw()
    fig.canvas.flush_events()



