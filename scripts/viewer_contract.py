"""Emit Python-owned viewer geometry and JSON schema, including factory defaults."""

import json

from tennis3d import schemas
from tennis3d.court.model import BALL_RADIUS_M, LANDMARK_NAMES, court_landmarks


def main():
    schema = schemas.Result.model_json_schema()
    models = {"Result": schema, **schema["$defs"]}
    for name, definition in models.items():
        model = getattr(schemas, name)
        for field_name, field in model.model_fields.items():
            if field.default_factory is not None:
                value = field.get_default(call_default_factory=True)
                if isinstance(value, schemas.StrictModel):
                    value = value.model_dump(mode="json")
                definition["properties"][field_name]["default"] = value
    print(
        json.dumps(
            {
                "schema": schema,
                "geometry": {
                    "points": court_landmarks().tolist(),
                    "names": LANDMARK_NAMES,
                    "radius": BALL_RADIUS_M,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
