# -*- coding: utf-8 -*-
#
# Copyright (c) 2018, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""Test the Moments method in cclib"""

from __future__ import print_function

import sys

import unittest
from unittest.mock import Mock

import numpy
from numpy.testing import assert_almost_equal

from cclib.method import Moments
from cclib.parser import GAMESS, Gaussian

sys.path.insert(1, "..")

from ..test_data import getdatafile


class MomentsTest(unittest.TestCase):
    def test_results(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate()
        assert_almost_equal(x[0], [0, 0, 0], 5)
        assert_almost_equal(x[1], [0, 0, -0.91543], 5)
        assert_almost_equal(x[2][0] + x[2][3] + x[2][5], 0)

    def test_origin_displacement(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate()
        y = Moments(data).calculate(origin=(2e7,18,28))
        assert_almost_equal(x[1], y[1])

    def test_origin_at_center_of_nuclear_charge(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate(origin="nuccharge")
        assert_almost_equal(x[0], [0, 0, 0], 6)

    def test_origin_at_center_of_mass(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate(origin="mass")
        assert_almost_equal(x[0], [0, 0, 0.0524806])

    def test_user_provided_origin(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate(origin=(1,1,1))
        assert_almost_equal(x[0], [1, 1, 1])
                            
    def test_user_provided_masses(self):
        data, _ = getdatafile(Gaussian, "basicGaussian16", ["water_mp2.log"])
        x = Moments(data).calculate(masses=[1,1,1], origin="mass")
        assert_almost_equal(x[0], [0, 0, -0.2780383])

    def test_results_storing(self):
        data, _ = getdatafile(GAMESS, "basicFirefly8.0", ["water_mp2.out"])
        m = Moments(data)
        m.calculate(population='mulliken')
        m.calculate(population='lowdin')
        assert 'mulliken' in m.results
        assert 'lowdin' in m.results

        
class TruncatedValuesTestHelper:
    def generate_data(state, atoms_num, decimals):
        charges = state.dirichlet(numpy.ones(atoms_num)) - 1 / atoms_num
                        
        coords = state.rand(atoms_num, 3)
        centroid = numpy.mean(coords, axis=0)
        transl_coords = coords - centroid

        k = 10**decimals
        trunc_coords = numpy.trunc(transl_coords * k) / k
        trunc_charges = numpy.trunc(charges * k) / k
            
        mock = Mock()
        mock.charge = 0
        mock.atomcharges = {'mulliken': trunc_charges}
        mock.atomcoords = trunc_coords.reshape(1, *coords.shape)
        mock.atomnos = numpy.ones(atoms_num)

        return mock

    def prepare_test(mock):
        def wrap(self):
            # For this mock, the center of nuclear charge is located at
            # (0,0,0).
            a = Moments(mock).calculate()
        
            msg = "Origin should located at zero point"
            assert_almost_equal(a[0], [0, 0, 0], 6, msg)

            # Place the origin somewhere inside the system of charges
            # and look at the dipole moment. It tests small origin
            # displacement.
            msg = "\mu(q=0) is invariant to the small origin displacement"
            b = Moments(mock).calculate(origin=mock.atomcoords[-1][0])
            assert_almost_equal(a[1], b[1], 6, msg)

            # It tests large origin displacement (whose values are larger
            # than accuracy of six decimals).
            msg = "\mu(q=0) is invariant to the large origin displacement"
            c = Moments(mock).calculate(origin=(1e7,2e7,3e7))
            assert_almost_equal(a[1], c[1], 6, msg)
            
        return wrap

    
class TruncatedValuesTest(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(unittest.makeSuite(MomentsTest))

    # TODO: After closing issue #455, implement this as parameterized
    # tests with the help of `pytest.mark.parametrize` decorator.
    for i in range(10):
        s = numpy.random.RandomState(i)
        atoms_num = s.randint(2, 20)
        mock = TruncatedValuesTestHelper.generate_data(s, atoms_num, 6)
        test_func = TruncatedValuesTestHelper.prepare_test(mock)
        setattr(TruncatedValuesTest, 'test_{}'.format(i), test_func)
    suite = unittest.makeSuite(TruncatedValuesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
