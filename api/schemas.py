"""Validação dos dados de entrada (regras de negócio da base ENB2012)."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Edificacao(BaseModel):
    compacidade_relativa: float = Field(..., ge=0.62, le=0.98, description="0,62 a 0,98")
    area_superficie: float = Field(..., ge=514.5, le=808.5, description="m², 514,5 a 808,5")
    area_parede: float = Field(..., ge=245, le=416.5, description="m², 245 a 416,5")
    area_telhado: float = Field(..., ge=110.25, le=220.5, description="m², 110,25 a 220,5")
    altura_total: Literal[3.5, 7.0] = Field(..., description="3,5 (1 andar) ou 7 (2 andares)")
    orientacao: Literal[2, 3, 4, 5] = Field(..., description="2=N, 3=L, 4=S, 5=O")
    area_janelas: Literal[0.0, 0.1, 0.25, 0.4] = Field(..., description="fração do piso")
    distribuicao_janelas: Literal[0, 1, 2, 3, 4, 5] = Field(..., description="0=sem janelas ... 5=oeste")

    @model_validator(mode="after")
    def janelas_coerentes(self):
        # na base, "sem janelas" (0) só existe junto com distribuição 0, e vice-versa
        if (self.area_janelas == 0) != (self.distribuicao_janelas == 0):
            raise ValueError("area_janelas = 0 exige distribuicao_janelas = 0 (e vice-versa)")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "compacidade_relativa": 0.90, "area_superficie": 563.5, "area_parede": 318.5,
                "area_telhado": 122.5, "altura_total": 7.0, "orientacao": 2,
                "area_janelas": 0.25, "distribuicao_janelas": 1,
            }
        }
    }


class Janela(BaseModel):
    lado: Literal["frente", "direita", "fundos", "esquerda"] = Field(..., description="Direita/esquerda de quem sai pela porta da frente")
    quantidade: int = Field(..., ge=1, le=50)
    largura_m: float = Field(..., gt=0, le=10)
    altura_m: float = Field(..., gt=0, le=5)


class Porta(BaseModel):
    lado: Literal["frente", "direita", "fundos", "esquerda"] = "frente"
    quantidade: int = Field(1, ge=1, le=20)
    largura_m: float = Field(0.8, gt=0, le=6)
    altura_m: float = Field(2.1, gt=0, le=4)
    material: Literal["madeira", "metal", "vidro"] = Field("madeira", description="Porta de vidro conta como vidro; as outras contam como parede")


class CasaDetalhada(BaseModel):
    """Imóvel descrito do jeito que a pessoa conhece: tipo, andares, formato, frente, tamanho, janelas e portas medidas."""
    tipo: Literal["casa", "apartamento"] = "casa"
    andares: Literal[1, 2] = Field(1, description="Só para casa")
    andar_apartamento: int = Field(1, ge=0, le=200, description="Só para apartamento: 0 = térreo")
    andares_predio: int = Field(1, ge=1, le=200, description="Só para apartamento: total de andares do prédio")
    lados_externos: list[Literal["frente", "direita", "fundos", "esquerda"]] = Field(
        default_factory=lambda: ["frente", "direita", "fundos", "esquerda"],
        description="Só para apartamento: lados com parede para fora do prédio")
    formato: int = Field(..., ge=0, le=5, description="0 = mais compacta ... 5 = mais alongada")
    orientacao: Literal[2, 3, 4, 5] = Field(..., description="Para onde a porta de entrada aponta: 2=N, 3=L, 4=S, 5=O")
    area_piso_m2: float = Field(..., ge=20, le=2000, description="Área de piso (m²), aparece no IPTU")
    janelas: list[Janela] = Field(default_factory=list)
    portas: list[Porta] = Field(default_factory=lambda: [Porta()])

    @model_validator(mode="after")
    def coerencia(self):
        if self.tipo == "apartamento":
            if self.andar_apartamento >= self.andares_predio:
                raise ValueError("O andar do apartamento precisa ser menor que o total de andares do prédio (térreo = 0)")
            if not self.lados_externos:
                raise ValueError("Pelo menos um lado do apartamento precisa dar para fora")
            fora = set(self.lados_externos)
            for j in self.janelas:
                if j.lado not in fora:
                    raise ValueError(f"Há janela na {j.lado}, mas esse lado não dá para fora do prédio")
        return self

    model_config = {"json_schema_extra": {"example": {
        "andares": 1, "formato": 2, "orientacao": 2, "area_piso_m2": 120,
        "janelas": [{"lado": "direita", "quantidade": 2, "largura_m": 2, "altura_m": 1.4},
                    {"lado": "frente", "quantidade": 1, "largura_m": 1.5, "altura_m": 1.2}]}}}


class Conta(BaseModel):
    """Dados para estimar a conta de luz em Goiás."""
    casa: Edificacao | None = None
    detalhada: CasaDetalhada | None = None
    cep: str = Field(..., pattern=r"^\d{5}-?\d{3}$", description="CEP do imóvel, ex.: 74000-000")
    area_climatizada_m2: float = Field(220.5, gt=0, le=2000, description="Área com ar-condicionado (m²). A base usa 220,5 m².")
    eficiencia_ar: float = Field(3.2, ge=1, le=8, description="Eficiência do ar-condicionado (W/W). Ver etiqueta do INMETRO.")
    usa_aquecimento: bool = Field(False, description="Usa aquecedor elétrico no inverno?")
    eficiencia_aquecedor: float = Field(1.0, ge=1, le=5, description="1 = aquecedor resistivo comum")
    outros_aparelhos_kwh_mes: float = Field(0, ge=0, le=5000, description="Geladeira, chuveiro, TV etc. (kWh/mês)")
    iluminacao_publica: float = Field(0, ge=0, le=500, description="Contribuição de iluminação pública (R$), está na conta")
    bandeira: Literal["verde", "amarela", "vermelha_1"] = "verde"
