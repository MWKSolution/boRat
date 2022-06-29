# boRat  

## Stress model for deviated borehole in anisotropic rock

---

![model_view](https://user-images.githubusercontent.com/105928466/176430528-c7a921c1-aee2-4572-9928-62037440f2e7.JPG)

---

Package for calculating the stress tensor around borehole wall for dipping anisotropic rocks and for arbitrary wellbore orientation.  
It uses ***Beltrami-Michell*** equations solutions for **transversely isotropic** (TIV), **orthotropic** (ORT) and **isotropic** (ISO) rocks.
For ISO rocks ***Kirsch*** equations could be used instead.  

Usage:

```python
    # initialize earth stress tensor instance:
    stress_pcs = Stress()
    # fill tensor with principal earth stresses:
    stress_pcs.set_from_PCS(SH=20, Sh=10, Sz=30, SHAzi=10)

    # define formation dip (dip and direction):
    dip = FormationDip(dip=15, dir=25)
    
    # define wellbore orietation along with mud pressure:
    wbo = WellboreOrientation(hazi=55, hdev=30, Pw=5)

    # define rock elastic parameters...
    # for isotropic rock:
    iso_rock = ISORock(E=30.14, PR=0.079)
    # for TIV rock:
    tiv_rock = TIVRock(Ev=15.42, Eh=31.17, PRv=0.32, PRhh=0.079, Gv=7.05)
    
    # make model of formation with drilled well:
    model = BoreholeModel(stress_pcs, iso_rock, dip, wbo, hoop_model='beltrami-michell')
    # for ISO rocks one can use 'kirsch' hoop model

    # to get stress tensor on borehole wall (hoop stress), in cylindrical coordinates counted from top of the hole (TOH) use:
    hoop = model.get_hoop_stress(theta)
    # ... where theta is &#x03B8; angle in cylindrical coordinates

    # to get 3D model view, unwrapped view of bedding planes intersecting borehole wall and stresses plot use: 
    model.show_all()
```

---

## Disclaimer !!!  

Project is in the development stage and don't pass all the tests...  
For example:  
For ISO rock, stresses from Beltrami-Michell and Kirsh should be the same for all model settings, but they are not...
```python
    ...
    modelK = BoreholeModel(stress_pcs, iso_rock, dip, wbo, hoop_model='kirsh')
    model.compare_stresses_with(modelK)
```

![compare](https://user-images.githubusercontent.com/105928466/176438356-f660e722-b05f-4e40-ab3d-35c6473e4aba.JPG)

Debugging is in progress...