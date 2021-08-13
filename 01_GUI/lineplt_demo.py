import psyplot.project as psy

axes = iter(psy.multiple_subplots(2, 2, n=3))
for var in ['t2m', 'u', 'v']:
    psy.plot.lineplot(
        'demo.nc',  # netCDF file storing the data
        name=var, # one plot for each variable
        t=range(5),  # one violin plot for each time step
        z=0, x=0,      # choose latitude and longitude as dimensions
        ylabel="{desc}",  # use the longname and units on the y-axis
        ax=next(axes),
        color='coolwarm', legend=False
    )
lines = psy.gcp(True)
lines.show()
