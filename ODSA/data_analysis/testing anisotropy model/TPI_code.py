import arcpy
from arcpy import env
from arcpy.sa import *

#Input elevation raster
dem = arcpy.GetParameterAsText(0)

#Area of specified neighborhood
neighborhood = arcpy.GetParameterAsText(1)

#TPI output raster dataset
#tpi_output = arcpy.GetParameterAsText(2)

#Standardized TPI output raster dataset
tpi_std_output = arcpy.GetParameterAsText(2)

arcpy.CheckOutExtension("Spatial")

#TPI compares the elevation of each cell in a mean elevation of a specified
#neighborhood around that cell.   
tpiCalculate = Int((dem - FocalStatistics(dem, neighborhood, "MEAN", "DATA")) + 0.5)

#Save TPI output
#tpiCalculate.save(tpi_output)

#Retrieve mean value
tpiCalculateMeanResult = arcpy.GetRasterProperties_management(tpiCalculate, "MEAN")

#Retrieve standard deviation value
tpiCalculateSTDResult = arcpy.GetRasterProperties_management(tpiCalculate, "STD")

#Assign mean and stdev to variables
tpiMean = (tpiCalculateMeanResult.getOutput(0))
tpiSTD = (tpiCalculateSTDResult.getOutput(0))

#Use raster calculator to calculate standardization
tpi_std = Int((arcpy.Raster(tpiCalculate) - float(tpiMean))/ float(tpiSTD))

#Save TPI mean and stdev values
tpi_std.save(tpi_std_output)

