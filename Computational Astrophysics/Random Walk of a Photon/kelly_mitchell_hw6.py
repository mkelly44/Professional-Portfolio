import numpy as np

# defining constants
c = 299792458.          # speed of light (m/s)
l = 4e-3                # random walk distance (m)
R = 0.7                 # radius of sun (m)
sec_in_year= 3.154e7 

# random walk coordinates from center of scatter location
def x_walk(ang):
    return l * np.cos(np.deg2rad(ang))
def y_walk(ang):
    return l * np.sin(np.deg2rad(ang))
def walk_radius(x,y):
    return np.sqrt(x**2 + y**2)

# initial displacement
x_dist = 0
y_dist = 0
scat_count = 0

# while loop until coordinates of photon are  beyond radius of sun
while(walk_radius(x_dist,y_dist) < R):
    phi = np.random.uniform(0,360)
    x_dist += x_walk(phi)
    y_dist += y_walk(phi)
    scat_count += 1

# scatter counter times l = total distance traveled, total distance traveled / speed of light = time and time / seconds in year is the conversion to time in years
# for each power of radius of the sun i.e. 0.7 to 7 to 70 is roughly 100 times the time and distance so (e9)**2 is the conversion for a radius of sun for 7e8 years

time_tot = scat_count * l / ( c * sec_in_year)
time_extrap = time_tot * 1e18

print("The total time is:",time_tot,"Years")
print("The total distance traveled is:",scat_count * l,"Meters")

print("The total extrapolated time is:",time_extrap,"Years")
print("The total extrapolated distance traveled is:",scat_count * l * 1e18, "Meters")
