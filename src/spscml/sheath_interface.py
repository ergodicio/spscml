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
    N: Differentiable[Float64] = Field(
            description="Plasma linear density; N_e = N_i = N [meters^-1]"
    )


class SheathOutputSchema(BaseModel):
    Ip: Differentiable[Float64] = Field(
            description="Plasma current [amperes]"
    )


