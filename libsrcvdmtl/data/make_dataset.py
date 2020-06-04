"""
Function that processes the shapefiles used in our models. Must be ran to generate the shape files under data/processed.
"""
import pandas as pd
import shapefile
import geopandas as gpd
from shutil import copyfile


def decode_field(f):
    return f.decode("ISO-8859-1") if type(f) == bytes else f


def make_shapes():
    """
    Function that processes the raw shape files in saves the filtered shape files under data/processed.
    :return: None
    """

    # Load the FSA (RTA) shapes
    shape_path = "../../data/external/fsa_shape/gfsa000b11a_e.shp"
    poly_shapefile = shapefile.Reader(shape_path)

    poly_df = pd.DataFrame(poly_shapefile.shapeRecords(), columns=["shapeRecords"])
    poly_df["shapes"] = poly_shapefile.shapes()
    poly_df['records'] = poly_df['shapeRecords'].map(lambda x: x.record)
    poly_df['poly_code'] = poly_df['shapeRecords'].map(lambda x: decode_field(x.record[0]))
    poly_df['poly_id'] = poly_df['shapeRecords'].map(lambda x: decode_field(x.record[1]))
    poly_df['poly_name'] = poly_df['shapeRecords'].map(lambda x: decode_field(x.record[2]))
    # Only keep FSA codes in Montreal
    poly_df = poly_df[poly_df['poly_code'].str.contains("^H([1-6]|[8-9])[A-Z]")].set_index("poly_code")

    shapewriter = shapefile.Writer(poly_shapefile.shapeType)
    shapewriter.fields = list(poly_shapefile.fields)
    shapewriter.records.extend(poly_df['records'])
    shapewriter._shapes.extend(poly_df['shapes'])

    # Save under data/processed.
    out_path = "../../data/processed/Montreal_fsa/Montreal_fsa.shp"
    shapewriter.save(out_path)

    copyfile(shape_path[:-4] + ".prj", out_path[:-4] + ".prj")

    # Dissemination areas (StatsCan)
    shape_path = "../../data/external/ilots_diffusion/gid_000b06a_f.shp"
    poly_shapefile = shapefile.Reader(shape_path)

    poly_df = pd.DataFrame(poly_shapefile.shapeRecords(), columns=["shapeRecords"])
    poly_df["shapes"] = poly_shapefile.shapes()
    poly_df['records'] = poly_df['shapeRecords'].map(lambda x: x.record)
    poly_df['poly_code'] = poly_df['shapeRecords'].map(lambda x: decode_field(x.record[0]))

    # code PR-DR-AD-ID
    # PR Quebec = 24
    # DR Montreal = 66
    # Filtering to keep codes in Montreal
    poly_df = poly_df[poly_df['poly_code'].str.contains("^2466")].set_index("poly_code")
    fields = [('DeletionFlag', 'C', 1, 0),
              ('IDIDU', 'C', 10, 0),
              ('PRLAT', 'N', 24, 15),
              ('PRLONG', 'N', 24, 15),
              ('ADIDU', 'C', 8, 0),
              ('SDRIDU', 'C', 7, 0),
              ('SRUIDU', 'C', 7, 0),
              ('DRIDU', 'C', 4, 0),
              ('RÉIDU', 'C', 4, 0),
              ('PRIDU', 'C', 2, 0),
              ('SRIDU', 'C', 10, 0),
              ('RMRIDU', 'C', 3, 0)]

    shapewriter = shapefile.Writer(poly_shapefile.shapeType)
    shapewriter.fields = list(fields)
    shapewriter.records.extend(poly_df['records'])
    shapewriter._shapes.extend(poly_df['shapes'])

    # Save to data/processeds
    out_path = "../../data/processed/Montreal_ilots/Montreal_ilots.shp"
    shapewriter.save(out_path)

    copyfile(shape_path[:-4] + ".prj", out_path[:-4] + ".prj")

    # Fire stations
    shape_path = "../../data/external/Secteur_casernes_operationnels/Secteur_de_casernes_opérationnels.shp"
    cas_shapes_ops = gpd.read_file(shape_path)
    # Convert coordinates to latitude and longitude
    cas_shapes_ops_proj = cas_shapes_ops.to_crs({'init': 'epsg:4326'})
    cas_shapes_ops_proj = cas_shapes_ops_proj[['NO_CAS_OP', 'OBJECTID', 'NOM_CAS_OP', 'ANGLE_R', 'ORIENTA', 'SCALE',
                                               'SHAPE_AREA', 'SHAPE_LEN', 'geometry']]
    # Save to data/processed
    out_path = "../../data/processed/Sect_cas_op/Sect_cas_op.shp"
    cas_shapes_ops_proj.to_file(out_path)

    # Grid shape files
    shape_path = "../../data/external/grid/grid_500m.shp"
    cas_shapes_ops = gpd.read_file(shape_path)
    # Convert coordinates to latitude and longitude
    cas_shapes_ops_proj = cas_shapes_ops.to_crs({'init': 'epsg:4326'})
    # Save to data/processed
    out_path = "../../data/processed/grid/grid_500m.shp"
    cas_shapes_ops_proj.to_file(out_path)

    return


def feature_rire(df, feature, save_path):
    """
    Generic function that is used to parse entries from the RIRE data set.
    :param df: (Pandas DataFrame) RIRE data set
    :param feature: (string) name of the feature to extract from the LIBELLE_CATEGORIE_IMMEUBLE column
    :param save_path: (string) path under which the parquet file will be saved
    :return:
    """
    df_feature = df[df["LIBELLE_CATEGORIE_IMMEUBLE"] == feature]
    df_feature["Longitude"] = df_feature["geometry"].apply(lambda x: x.centroid.coords.xy[0][0])
    df_feature["Latitude"] = df_feature["geometry"].apply(lambda x: x.centroid.coords.xy[1][0])
    df_feature[["SUPERFICIE_BATIMENT", "SUPERFICIE_TERRAIN", "Latitude", "Longitude"]].reset_index(
        drop=True).to_parquet(save_path)


def process_rire():
    """
    Function that extracts the useful information for the first responders project from the raw RIRE GeoJSON file. We
    divide it in three parts: schools, factories, parks and senior homes. Four different parquet files are generated and
    saved under data/processed.
    :return: None
    """
    fname = "../../data/external/matrice_graphique_extract.geojson"
    df = gpd.read_file(fname)
    feature_rire(df, "Ecoles, colleges, universites et autres du reseau de l'education",
                 "../../data/processed/ecoles.parquet")
    feature_rire(df, "Usines", "../../data/processed/usines.parquet")
    feature_rire(df, "Parcs", "../../data/processed/parcs.parquet")
    feature_rire(df, "Residence personnes agees", "../../data/processed/residences_personnes_agees.parquet")
    return


if __name__ == "__main__":
    # Filter the shapes
    make_shapes()
    # Filter the RIRE data set
    process_rire()
