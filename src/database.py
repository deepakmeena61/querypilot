"""
Database setup: reads dataset.csv and normalizes into relational DuckDB tables.
Run this script once before starting the app: python -m src.database
"""

import json
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "dataset.csv"
DB_PATH = DATA_DIR / "spotify.duckdb"
SCHEMA_PATH = DATA_DIR / "schema.json"

KEY_NAMES = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
             6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}


def load_and_clean_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0)
    before = len(df)
    df = df.dropna(subset=["track_name", "artists"])
    df = df.drop_duplicates(subset=["track_id"])
    after = len(df)
    logger.info(f"Cleaned CSV: {before} → {after} rows ({before - after} dropped)")
    return df


def build_tables(df: pd.DataFrame) -> dict:
    """Build normalized tables from the flat CSV dataframe."""

    # albums
    albums = (
        pd.DataFrame({"album_name": df["album_name"].unique()})
        .reset_index(drop=True)
    )
    albums.insert(0, "album_id", range(1, len(albums) + 1))

    # artists (split semicolons)
    all_artists = set()
    for raw in df["artists"].dropna():
        for a in raw.split(";"):
            all_artists.add(a.strip())
    artists = pd.DataFrame({"artist_name": sorted(all_artists)}).reset_index(drop=True)
    artists.insert(0, "artist_id", range(1, len(artists) + 1))

    # genres
    genres = (
        pd.DataFrame({"genre_name": df["track_genre"].unique()})
        .reset_index(drop=True)
    )
    genres.insert(0, "genre_id", range(1, len(genres) + 1))

    # key_names lookup
    key_names_df = pd.DataFrame(
        [{"key": k, "key_name": v} for k, v in KEY_NAMES.items()]
    )

    # tracks (with FK lookups)
    album_map = dict(zip(albums["album_name"], albums["album_id"]))
    tracks = df[
        ["track_id", "track_name", "album_name", "popularity", "duration_ms",
         "explicit", "danceability", "energy", "key", "loudness", "mode",
         "speechiness", "acousticness", "instrumentalness", "liveness",
         "valence", "tempo", "time_signature"]
    ].copy()
    tracks["album_id"] = tracks["album_name"].map(album_map)
    tracks["duration_min"] = (tracks["duration_ms"] / 60000).round(2)
    tracks = tracks.drop(columns=["album_name"])

    # track_artists junction
    artist_map = dict(zip(artists["artist_name"], artists["artist_id"]))
    ta_rows = []
    for _, row in df[["track_id", "artists"]].iterrows():
        for a in str(row["artists"]).split(";"):
            name = a.strip()
            if name in artist_map:
                ta_rows.append({"track_id": row["track_id"], "artist_id": artist_map[name]})
    track_artists = pd.DataFrame(ta_rows).drop_duplicates()

    # track_genres junction
    genre_map = dict(zip(genres["genre_name"], genres["genre_id"]))
    track_genres = df[["track_id", "track_genre"]].copy()
    track_genres["genre_id"] = track_genres["track_genre"].map(genre_map)
    track_genres = track_genres[["track_id", "genre_id"]].drop_duplicates()

    return {
        "albums": albums,
        "artists": artists,
        "genres": genres,
        "key_names": key_names_df,
        "tracks": tracks,
        "track_artists": track_artists,
        "track_genres": track_genres,
    }


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW v_track_details AS
        SELECT
            t.track_id,
            t.track_name,
            al.album_name,
            t.popularity,
            t.duration_ms,
            t.duration_min,
            t.explicit,
            t.danceability,
            t.energy,
            t.key,
            kn.key_name,
            t.loudness,
            t.mode,
            CASE t.mode WHEN 1 THEN 'Major' ELSE 'Minor' END AS mode_name,
            t.speechiness,
            t.acousticness,
            t.instrumentalness,
            t.liveness,
            t.valence,
            t.tempo,
            t.time_signature,
            g.genre_name
        FROM tracks t
        JOIN albums al ON t.album_id = al.album_id
        JOIN track_genres tg ON t.track_id = tg.track_id
        JOIN genres g ON tg.genre_id = g.genre_id
        LEFT JOIN key_names kn ON t.key = kn.key
    """)

    con.execute("""
        CREATE OR REPLACE VIEW v_artist_tracks AS
        SELECT
            ar.artist_name,
            t.track_id,
            t.track_name,
            al.album_name,
            t.popularity,
            g.genre_name,
            t.danceability,
            t.energy,
            t.valence,
            t.tempo,
            t.duration_min
        FROM tracks t
        JOIN track_artists ta ON t.track_id = ta.track_id
        JOIN artists ar ON ta.artist_id = ar.artist_id
        JOIN albums al ON t.album_id = al.album_id
        JOIN track_genres tg ON t.track_id = tg.track_id
        JOIN genres g ON tg.genre_id = g.genre_id
    """)


COLUMN_DESCRIPTIONS = {
    "track_id": "Unique Spotify track identifier",
    "track_name": "Name of the song/track",
    "album_name": "Name of the album containing the track",
    "album_id": "Foreign key reference to the albums table",
    "artist_id": "Auto-incrementing unique identifier for each artist",
    "artist_name": "Name of the artist or band",
    "genre_id": "Auto-incrementing unique identifier for each genre",
    "genre_name": "Music genre label (e.g., pop, rock, hip-hop)",
    "popularity": "Popularity score from 0-100 (higher = more popular on Spotify)",
    "duration_ms": "Track duration in milliseconds",
    "duration_min": "Track duration in minutes (computed from duration_ms)",
    "explicit": "Whether the track contains explicit lyrics (True/False)",
    "danceability": "How suitable a track is for dancing: 0.0=least danceable, 1.0=most danceable",
    "energy": "Perceptual measure of intensity and activity: 0.0=calm, 1.0=energetic",
    "key": "Musical key the track is in (0=C, 1=C#, 2=D, ..., 11=B)",
    "key_name": "Human-readable musical key name (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)",
    "loudness": "Overall loudness in decibels (dB), typically -60 to 0; higher=louder",
    "mode": "Musical mode: 1=Major (often brighter/happier), 0=Minor (often darker/sadder)",
    "mode_name": "Human-readable mode name: Major or Minor",
    "speechiness": "Presence of spoken words: 0.0=no speech, 1.0=pure spoken word",
    "acousticness": "Confidence the track is acoustic: 0.0=not acoustic, 1.0=definitely acoustic",
    "instrumentalness": "Likelihood the track has no vocals: 0.0=vocals present, 1.0=instrumental",
    "liveness": "Probability the track was recorded live: >0.8=likely live performance",
    "valence": "Musical positiveness: 0.0=sad/negative, 1.0=happy/positive/euphoric",
    "tempo": "Estimated tempo in beats per minute (BPM)",
    "time_signature": "Estimated time signature beats per bar (3 to 7, e.g., 4=4/4 time)",
}


def generate_schema_json(tables: dict, con: duckdb.DuckDBPyConnection) -> dict:
    schema = {"tables": {}, "views": {}, "relationships": []}

    for table_name, df in tables.items():
        cols = {}
        for col in df.columns:
            sample_vals = df[col].dropna().head(3).tolist()
            cols[col] = {
                "type": str(df[col].dtype),
                "description": COLUMN_DESCRIPTIONS.get(col, ""),
                "sample_values": sample_vals,
            }
        schema["tables"][table_name] = {
            "description": _table_description(table_name),
            "row_count": len(df),
            "columns": cols,
        }

    # Views
    for view_name in ["v_track_details", "v_artist_tracks"]:
        try:
            sample = con.execute(f"SELECT * FROM {view_name} LIMIT 3").df()
            cols = {}
            for col in sample.columns:
                sample_vals = sample[col].dropna().head(3).tolist()
                cols[col] = {
                    "description": COLUMN_DESCRIPTIONS.get(col, ""),
                    "sample_values": sample_vals,
                }
            schema["views"][view_name] = {
                "description": _view_description(view_name),
                "columns": cols,
            }
        except Exception as e:
            logger.warning(f"Could not sample view {view_name}: {e}")

    schema["relationships"] = [
        {"from": "tracks.album_id", "to": "albums.album_id"},
        {"from": "track_artists.track_id", "to": "tracks.track_id"},
        {"from": "track_artists.artist_id", "to": "artists.artist_id"},
        {"from": "track_genres.track_id", "to": "tracks.track_id"},
        {"from": "track_genres.genre_id", "to": "genres.genre_id"},
        {"from": "tracks.key", "to": "key_names.key"},
    ]
    return schema


def _table_description(name: str) -> str:
    descriptions = {
        "tracks": "Core table of 114K Spotify tracks with audio features and popularity scores",
        "artists": "Unique artists extracted from the dataset; each artist has a unique ID",
        "albums": "Unique albums extracted from the dataset; each album has a unique ID",
        "genres": "Unique music genres in the dataset (e.g., pop, rock, hip-hop, classical)",
        "key_names": "Lookup table mapping integer key values (0-11) to musical key names (C through B)",
        "track_artists": "Junction table: many-to-many relationship between tracks and artists",
        "track_genres": "Junction table: one genre per track in this dataset",
    }
    return descriptions.get(name, "")


def _view_description(name: str) -> str:
    descriptions = {
        "v_track_details": (
            "Pre-joined analytical view: tracks + albums + genres + key_names. "
            "Use this for most queries about track attributes, audio features, and genres. "
            "Does NOT include artist info — use v_artist_tracks for artist queries."
        ),
        "v_artist_tracks": (
            "Pre-joined analytical view: tracks + artists + albums + genres. "
            "Use this for queries involving artist names, artist statistics, or filtering by artist."
        ),
    }
    return descriptions.get(name, "")


def setup_database(force: bool = False) -> None:
    """Main entry point: build the normalized DuckDB database from the CSV."""
    if DB_PATH.exists() and not force:
        logger.info(f"Database already exists at {DB_PATH}. Use force=True to rebuild.")
        return

    logger.info(f"Loading CSV from {CSV_PATH}...")
    df = load_and_clean_csv()

    logger.info("Normalizing into relational tables...")
    tables = build_tables(df)

    logger.info(f"Writing to DuckDB at {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    try:
        for table_name, table_df in tables.items():
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.register("_tmp", table_df)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp")
            logger.info(f"  {table_name}: {len(table_df):,} rows")

        logger.info("Creating views...")
        create_views(con)

        logger.info("Generating schema.json...")
        schema = generate_schema_json(tables, con)
        with open(SCHEMA_PATH, "w") as f:
            json.dump(schema, f, indent=2, default=str)

        # Summary stats
        print("\n=== QueryPilot Database Summary ===")
        for table_name in ["tracks", "artists", "albums", "genres", "track_artists", "track_genres"]:
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  {table_name:20s}: {count:>8,} rows")
        print("===================================\n")

    finally:
        con.close()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a read-only connection to the database. Creates DB if missing."""
    if not DB_PATH.exists():
        setup_database()
    return duckdb.connect(str(DB_PATH), read_only=False)


def load_schema() -> dict:
    """Load the schema JSON file."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema file not found at {SCHEMA_PATH}. Run setup_database() first."
        )
    with open(SCHEMA_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    setup_database(force=True)
