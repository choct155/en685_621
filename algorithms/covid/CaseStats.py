from dataclasses import dataclass
from numbers import Rational
from typing import NamedTuple

@dataclass
class CaseStats:
    """
    Struct holding relevant information that can be cumulatively extracted
    from a series of cumulative COVID-19 cases within an abritrary cell (e.g
    city, state, or country).

    Keyword arguments:
    stock      -- Cumalitive count of cases
    flow       -- Daily new cases
    max_flow   -- Maximum count of new cases in a day for the days since onset
    avg_flow   -- Average count of new cases in a day for the days since onset
    onset_days -- Days since the first case (inclusive of the first day)
    """
    stock: Rational
    flow: Rational
    max_flow: Rational
    avg_flow: Rational
    onset_days: int

@dataclass
class CaseStatsByGeo:
    """
    Struct holding relevant information that can be cumulatively extracted
    from a series of cumulative COVID-19 cases within an abritrary cell (e.g
    city, state, or country). Distinguished from CaseStats because of the
    desire to hold geographic labels in some cases.

    Keyword arguments:
    geo        -- Name of the geographic area in which the cases occurred
    stock      -- Cumalitive count of cases
    flow       -- Daily new cases
    max_flow   -- Maximum count of new cases in a day for the days since onset
    avg_flow   -- Average count of new cases in a day for the days since onset
    onset_days -- Days since the first case (inclusive of the first day)
    """
    geo: str
    stock: Rational
    flow: Rational
    max_flow: Rational
    avg_flow: Rational
    onset_days: int

class CaseStatsTup(NamedTuple):
    """Just to play nice with pandas"""
    stock: Rational
    flow: Rational
    max_flow: Rational
    avg_flow: Rational
    onset_days: int

class CaseStatsByGeoTup(NamedTuple):
    """Just to play nice with pandas"""
    geo: str
    stock: Rational
    flow: Rational
    max_flow: Rational
    avg_flow: Rational
    onset_days: int

class CaseStatOps:

    @staticmethod
    def toCaseStatsTup(cs: CaseStats) -> CaseStatsTup:
        return CaseStatsTup(cs.stock, cs.flow, cs.max_flow, cs.avg_flow, cs.onset_days)

    @staticmethod
    def toCaseStatsByGeoTup(cs: CaseStatsByGeo) -> CaseStatsByGeoTup:
        return CaseStatsByGeoTup(cs.geo, cs.stock, cs.flow, cs.max_flow, cs.avg_flow, cs.onset_days)