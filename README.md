# ICP: Iterative Closest Point

Python implementation of m-dimensional Iterative Closest Point method.  ICP finds a best fit rigid body transformation between two point sets.  Correspondence between the points is not assumed. Included is an SVD-based least-squared best-fit algorithm for corresponding point sets.

This repository was initially created as a fork of [ClayFlanningan/icp](https://github.com/ClayFlannigan/icp). Many thanks to him and contributors for their work who helped me create my own version of this algorithm.

## Dependencies

```
pip install -r requirements.txt
```

## Study references

- [Pyoints: ICP for point cloud alignment](https://laempy.github.io/pyoints/tutorials/icp.html)
- [Open3D: ICP registration](http://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html)
