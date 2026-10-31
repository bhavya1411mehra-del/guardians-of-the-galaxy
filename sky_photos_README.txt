SKY PHOTO TRAINING DATA INSTRUCTIONS

Create your own folder sky_photos beside app.py. Inside add: clear/, cloudy/, fog/, haze/.
Put consented, properly licensed photographs into each directory. At least 5 per class,
but dozens/hundreds of diverse images are better. Run:

py -m pip install -r requirements-vision.txt
py train_sky.py --data sky_photos --output sky_prototypes.npz

This creates an optional data-trained class prototype model. It does not measure PM2.5.
