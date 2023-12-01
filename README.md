# Modified Void Descriptor Function (mVDF)

The modified void descriptor function (VDF) that improves upon pore-related metrics commonly reported in the literature by simultaneously accounting for pore sizes, pore shapes, pore clustering, pore-pore interaction, and pore locations relative to free surfaces of the specimen.

The modified VDF is a tool written in Python that is able to calculate the VDF value given a pore network file and some parameters pertaining to the specimen.

If you use this tool or its key elements in your publication, we kindly ask you to cite the following publication:
```
D.S. Watring, J.T. Benzing, O.L. Kafka, L.A. Liew, N.H. Moser, J. Erickson, N. Hrabe, & A.D. Spear. "Evaluation of a modified void descriptor function to uniquely characterize pore networks and predict fracture-related properties in additively manufactured metals" _Submitted Acta Materialia_
```

# Prerequisites
For this code to work, you will need the following:
* Python
* Numpy
* Scikit-Learn
* Matplotlib


# Running example pore file
To run the example pore file:
* Import pore file and calculate the vdf: network_VDF = VDF(network_path, alpha, rho, gamma, zeta)


# Running your own pore file
To run your own pore file:
* Ensure you pore file aligns with example pore file (see PoreFileReadMe)
* Z axis must be the axis aligned with gauge length
* Ensure consistant units (e.g., X, Y, Z, Volume, etc.)
