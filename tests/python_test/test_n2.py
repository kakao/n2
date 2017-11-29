# Copyright (c) 2017 Kakao corp
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This unittests are based on
# https://github.com/spotify/annoy/blob/master/test/annoy_test.py

import unittest
import random
import os

try:
    xrange
except NameError:
    xrange = range

from n2 import HnswIndex


class TestCase(unittest.TestCase):
    def assertAlmostEqual(self, x, y):
        super(TestCase, self).assertAlmostEqual(x, y, 3)


class AngularTest(TestCase):
    def test_search_by_vector(self):
        f = 3
        i = HnswIndex(f)
        i.add_data([0, 0, 1])
        i.add_data([0, 1, 0])
        i.add_data([1, 0, 0])
        i.build(max_m0=10, m=5)

        self.assertEqual(i.search_by_vector([3, 2, 1], 3), [0, 1, 2])
        self.assertEqual(i.search_by_vector([1, 2, 3], 3), [0, 1, 2])
        self.assertEqual(i.search_by_vector([2, 0, 1], 3), [0, 1, 2])

    def test_search_by_id(self):
        f = 3
        i = HnswIndex(f)
        i.add_data([2, 1, 0])
        i.add_data([1, 2, 0])
        i.add_data([0, 0, 1])
        i.build(max_m0=10)

        self.assertEqual(i.search_by_id(0, 3), [0, 1, 2])
        self.assertEqual(i.search_by_id(1, 3), [1, 0, 2])
        self.assertTrue(i.search_by_id(2, 3) in [[2, 0, 1], [2, 1, 0]])  # could be either

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in xrange(n_rounds):
            # create random points at distance x from (1000, 0, 0, ...)
            f = 10
            i = HnswIndex(f, 'L2')
            for j in xrange(n_points):
                p = [random.gauss(0, 1) for z in xrange(f - 1)]
                norm = sum([pi ** 2 for pi in p]) ** 0.5
                x = [1000] + [pi / norm * j for pi in p]
                i.add_data(x)

            i.build()

            nns = i.search_by_vector([1000] + [0] * (f-1), n)
            self.assertEqual(nns, sorted(nns))  # should be in order
            # The number of gaps should be equal to the last item minus n-1
            found += len([_x for _x in nns if _x < n])

        return 1.0 * found / (n * n_rounds)

    def test_precision_1(self):
        self.assertTrue(self.precision(1) >= 0.98)

    def test_precision_10(self):
        self.assertTrue(self.precision(10) >= 0.98)

    def test_precision_100(self):
        self.assertTrue(self.precision(100) >= 0.98)

    def test_precision_1000(self):
        self.assertTrue(self.precision(1000) >= 0.98)


class L2Test(TestCase):
    def test_search_by_vector(self):
        f = 2
        i = HnswIndex(f, 'L2')
        i.add_data([2, 2])
        i.add_data([3, 2])
        i.add_data([3, 3])
        i.build()

        self.assertEqual(i.search_by_vector([4, 4], 3), [2, 1, 0])
        self.assertEqual(i.search_by_vector([1, 1], 3), [0, 1, 2])
        self.assertEqual(i.search_by_vector([4, 2], 3), [1, 2, 0])

    def test_search_by_id(self):
        f = 2
        i = HnswIndex(f, 'L2')
        i.add_data([2, 2])
        i.add_data([3, 2])
        i.add_data([3, 3])
        i.build()

        self.assertEqual(i.search_by_id(0, 3), [0, 1, 2])
        self.assertEqual(i.search_by_id(2, 3), [2, 1, 0])

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        # q = [random.gauss(0, 10) for z in xrange(f)]
        i = HnswIndex(f, 'L2')
        for j in xrange(0, 10000, 2):
            p = [random.gauss(0, 1) for z in xrange(f)]
            x = [1 + pi + random.gauss(0, 1e-2) for pi in p]  # todo: should be q[i]
            y = [1 + pi + random.gauss(0, 1e-2) for pi in p]
            i.add_data(x)
            i.add_data(y)

        i.build()
        for j in xrange(0, 10000, 2):
            self.assertEqual(i.search_by_id(j, 2), [j, j+1])
            self.assertEqual(i.search_by_id(j+1, 2), [j+1, j])

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in xrange(n_rounds):
            # create random points at distance x
            f = 10
            i = HnswIndex(f, 'L2')
            for j in xrange(n_points):
                p = [random.gauss(0, 1) for z in xrange(f)]
                norm = sum([pi ** 2 for pi in p]) ** 0.5
                x = [pi / norm * j for pi in p]
                i.add_data(x)

            i.build()

            nns = i.search_by_vector([0] * f, n)
            self.assertEqual(nns, sorted(nns))  # should be in order
            # The number of gaps should be equal to the last item minus n-1
            found += len([_x for _x in nns if _x < n])

        return 1.0 * found / (n * n_rounds)

    def test_precision_1(self):
        self.assertTrue(self.precision(1) >= 0.98)

    def test_precision_10(self):
        self.assertTrue(self.precision(10) >= 0.98)

    def test_precision_100(self):
        self.assertTrue(self.precision(100) >= 0.98)

    def test_precision_1000(self):
        self.assertTrue(self.precision(1000) >= 0.98)


class BasicTest(TestCase):
    dim = 100
    model_fname = 'dummy.hnsw'

    @classmethod
    def setUpClass(self):
        index = HnswIndex(self.dim)
        for i in xrange(1000):
            v = [random.gauss(0, 1) for z in xrange(self.dim)]
            index.add_data(v)
        index.build(n_threads=12)
        index.save(self.model_fname)

    @classmethod
    def tearDownClass(self):
        os.remove(self.model_fname)

    def test01_small_invalid_dimension(self):
        index = HnswIndex(30)
        this_is_abnormal = False
        try:
            index.load(self.model_fname)
            this_is_abnormal = True
        except:
            pass
        finally:
            del index
        self.assertFalse(this_is_abnormal)

    def test02_small_invalid_dimension2(self):
        index = HnswIndex(80)
        this_is_abnormal = False
        try:
            v = [random.gauss(0, 1) for z in xrange(100)]
            index.add_data(v)
            this_is_abnormal = True
        except:
            pass
        finally:
            del index
        self.assertFalse(this_is_abnormal)

    def test03_small_add_data_after_loading(self):
        index = HnswIndex(self.dim)
        index.load(self.model_fname)
        this_is_abnormal = False
        try:
            v = [random.gauss(0, 1) for z in xrange(self.dim)]
            index.add_data(v)
            this_is_abnormal = True
        except:
            pass
        finally:
            del index
        self.assertFalse(this_is_abnormal)
