import sys

def test_run_cpp_tests():
    import cppimport
    test_main = cppimport.imp('test_main')
    test_main.run_tests(sys.argv)
