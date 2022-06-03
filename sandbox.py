import numpy as np
from mantid.simpleapi import *
import matplotlib.pyplot as plt

print(mtd.getObjectNames())
ReflectometryISISLoadAndProcess(InputRunList=str("66233"), AnalysisMode='MultiDetectorAnalysis',
                                        ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_'+str(66233))

print(mtd.getObjectNames())