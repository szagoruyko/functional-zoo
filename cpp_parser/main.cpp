#include <string>
#include <vector>
#include <iostream>

#include "H5Cpp.h"

const H5std_string ROOT_GROUP_NAME("/data_0");

int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "[ERROR] Not enough arguments.";
		std::cerr << "Please provide the path to the HDF5 file." << std::endl;
		return 0;
	}

	const H5std_string FILE_NAME(argv[1]);

	H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
	H5::Group root_grp = file.openGroup(ROOT_GROUP_NAME);

	for (size_t i = 0; i < root_grp.getNumObjs(); ++i) {
		const H5std_string tensor_name(root_grp.getObjnameByIdx(i));
		const H5std_string grp_name(ROOT_GROUP_NAME + "/"+ tensor_name);
		const H5std_string DATASET_NAME(grp_name + "/data_0");

		// open data set
		H5::DataSet dset = file.openDataSet(DATASET_NAME);

		// check tensor rank
		H5::DataSpace dspace = dset.getSpace();
		hsize_t rank = dspace.getSimpleExtentNdims();

		// check tensor dimensions
		hsize_t dims[rank];
		dspace.getSimpleExtentDims(dims, nullptr);

		// check tensor total size
		hsize_t t_size = dspace.getSimpleExtentNpoints();

		// fill Tensor
		// TODO: handle different data types
		std::vector<double> tensor(t_size);
		dset.read(tensor.data(), H5::PredType::NATIVE_DOUBLE, dspace);

		// LOG Tensor info

		std::cout << "-- TENSOR " << std::endl;
		std::cout << "   Name: " << tensor_name << std::endl;
		std::cout << "   Rank: " << rank << std::endl;
		std::cout << "   Dims: ";
		for (size_t j = 0; j < rank; ++j) {
			std::cout << dims[j] << " ";
		}
		std::cout << std::endl << std::endl;
	}

	return 0;
}
