from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64


class SheathInputSchema(BaseModel):
    Vp: Differentiable[Float64] = Field(
            description="Anode-cathode voltage gap [volts]"
    )
    T: Differentiable[Float64] = Field(
            description="Plasma temperature [eV]"
    )
    n: Differentiable[Float64] = Field(
            description="Plasma volumetric density; n_e = n_i = n [meters^-3]"
    )
    Lz: Differentiable[Float64] = Field(
            description="Inter-electrode length [meters]"
    )
    mlflow_parent_run_id: str | None = Field(
            default=None,
            description="The parent mlflow run id, if any"
    )



class SheathOutputSchema(BaseModel):
    j: Differentiable[Float64] = Field(
            description="Plasma current density [amperes / m^2]"
    )


