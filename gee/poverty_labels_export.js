// ==============================
// 1. DEFINE AOI
// ==============================
var aoi = ee.Geometry.Point([90.4003, 23.7806]).buffer(60000);

// ==============================
// 2. SENTINEL-2 IMAGE (WITH SWIR)
// ==============================
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .clip(aoi);

var image = s2.select(['B4','B3','B2','B8','B11']);

// ==============================
// 3. METAL ROOF INDEX
// ==============================
var NIR  = s2.select('B8');
var SWIR = s2.select('B11');
var metalRoofIndex = SWIR.divide(NIR).rename('metal_roof');
var tinRoofMask = metalRoofIndex.gt(1.0);

// ==============================
// 4. CREATE GRID
// ==============================
var grid = aoi.coveringGrid(aoi.projection(), 800);

// ==============================
// 5. POVERTY SCORE PER GRID CELL
// ==============================
var labeledGrid = grid.map(function(cell) {
  var geom = cell.geometry();
  var tinCount = tinRoofMask.sample({
    region: geom,
    scale: 10,
    numPixels: 500,
    geometries: false
  }).aggregate_sum('metal_roof');

  var totalCount = tinRoofMask.sample({
    region: geom,
    scale: 10,
    numPixels: 500,
    geometries: false
  }).size();

  var score = ee.Number(tinCount).divide(ee.Number(totalCount));
  return cell.set('poverty_score', score);
});

// ==============================
// 6. EXPORT LABELS CSV
// ==============================
Export.table.toDrive({
  collection: labeledGrid,
  description: 'bracu_labels_roofs4',
  folder: 'BRACU_DATASET',
  fileFormat: 'CSV'
});

// ==============================
// 7. EXPORT FULL IMAGE
// ==============================
Export.image.toDrive({
  image: image,
  description: 'bracu_full_image4',
  folder: 'BRACU_DATASET',
  fileNamePrefix: 'bracu_full_image4',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
