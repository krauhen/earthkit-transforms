Spatial aggregations, masking and neighbourhood operators
---------------------------------------------------------

The :doc:`../autodocs/earthkit.transforms.spatial` module includes methods
for aggregating data in space. This includes masking and aggregating data with geometries,
and neighbourhood operators on regular grids.

To mask data you can use the :func:`mask` function. This function takes an xarray
data object and a geometry object and returns the data object with the values
outside the geometry masked.

.. dropdown:: Show API documentation for ``mask``

   .. autofunction:: earthkit.transforms.spatial.mask
      :no-index:


To calculate an aggregated value for the data within a geometry you can use the
:func:`reduce` function. This function takes an xarray data object and a geometry
object and returns the aggregated value of the data within the geometry. The `how`
parameter can be used to specify the aggregation method. The default is `mean`.

.. dropdown:: Show API documentation for ``reduce``

   .. autofunction:: earthkit.transforms.spatial.reduce
      :no-index:


Using a bounding box instead of a geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both :func:`mask` and :func:`reduce` accept an ``area`` keyword argument as an
alternative to providing a ``geodataframe``.  ``area`` is a dictionary with keys
``"north"``, ``"south"``, ``"east"`` and ``"west"`` that defines a simple
bounding box:

.. code-block:: python

    ekt.spatial.reduce(
        ds, area={"north": 60, "south": 30, "east": 40, "west": -10}, how="mean"
    )

The bounding box is converted internally to a single-polygon GeoDataFrame.
Providing both ``area`` and ``geodataframe`` raises a ``ValueError``.

.. note::
   Areas that cross the anti-meridian (where ``west > east``) are not currently
   supported.


Neighbourhood operators
^^^^^^^^^^^^^^^^^^^^^^^

To smooth data on a regular latitude/longitude grid you can use :func:`disk_mean`. For
each grid cell it returns the mean of the defined values within a disk of a given radius,
measured in grid cells. Missing values are left out of the mean.

.. dropdown:: Show API documentation for ``disk_mean``

   .. autofunction:: earthkit.transforms.spatial.disk_mean
      :no-index:

To fill missing (NaN) cells from their neighbours you can use :func:`fill_missing`. Each
missing cell is replaced by a distance-weighted mean of the defined cells within a disk,
provided that not too many of the cells in the disk are missing.

.. dropdown:: Show API documentation for ``fill_missing``

   .. autofunction:: earthkit.transforms.spatial.fill_missing
      :no-index:


Specifying the spatial variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The spatial methods (:func:`mask`, :func:`reduce`, :func:`disk_mean` and :func:`fill_missing`) operate over the latitude and
longitude coordinates of the data object. By default these coordinates are detected
automatically from the metadata of the data object, so in most cases you do not need to
provide them explicitly. The detection looks, in order, for:

1. A coordinate whose ``axis`` attribute is ``"Y"`` (latitude) or ``"X"`` (longitude).
2. A coordinate whose ``standard_name`` attribute is a CF spatial name, e.g.
   ``"latitude"``/``"grid_latitude"`` or ``"longitude"``/``"grid_longitude"``.
3. A coordinate whose name matches one of the recognised names, i.e. ``"latitude"`` or
   ``"lat"`` for latitude, and ``"longitude"``, ``"long"`` or ``"lon"`` for longitude.

If the coordinates cannot be detected automatically - for example when your data uses
non-standard names - or if the automatic detection selects the wrong coordinate, you can
override it using the ``lat_key`` and ``lon_key`` keyword arguments:

.. code-block:: python

    ekt.spatial.reduce(
        ds,
        area={"north": 60, "south": 30, "east": 40, "west": -10},
        lat_key="y_coord",
        lon_key="x_coord",
    )

Both ``lat_key`` and ``lon_key`` are accepted by :func:`mask`, :func:`reduce`, :func:`disk_mean` and :func:`fill_missing`. A worked
example is available in the how-to guide
:doc:`../how-tos/spatial/howto_specify_lat_lon_keys`.


In addition to the above functions, the spatial module also includes several methods
for computing the intermedieate steps of the aggregation process. These methods are
documented in the API reference guide: :doc:`../autodocs/earthkit.transforms.spatial`
