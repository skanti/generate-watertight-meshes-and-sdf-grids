#include <cmath>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Vox.h"
#include "voxelize_project.h"
#include "LoaderMesh.h"
#include "watershed.cuh"
#include "sdf.cuh"
#include <cxxopts.hpp>


struct Args {
	std::string filename_in;
	std::string filename_out;
	float res;
	int trunc;
	int pad;
  bool normalize;
	bool verbose;
} args;


class World {
	public:
		typedef uint8_t vtype;

		void run() {

			load_mesh(args.filename_in, mesh);
      if (args.normalize) {
        Eigen::Vector3f vmin = mesh.V.rowwise().minCoeff();
        Eigen::Vector3f vmax = mesh.V.rowwise().maxCoeff();

        Eigen::Vector3f center = 0.5*(vmax + vmin);
        Eigen::Vector3f extent = (vmax - vmin);

        mesh.V = mesh.V.colwise() - center;
        mesh.V = mesh.V.array().colwise() / extent.array();


      }
			auto vox = voxelize<vtype>(mesh.V, mesh.F, args.res, ID_SOLID, ID_WALL, args.pad, args.verbose);
			Eigen::Array3i dims = vox.dims;
			int n_elems = dims(0)*dims(1)*dims(2);

			Vox<float> voxf;
			voxf.dims = vox.dims;
			voxf.res = vox.res;
			voxf.grid2world = vox.grid2world;
			voxf.sdf.resize(n_elems);

			std::vector<vtype> water = vox.sdf;
			int n_empty0 = std::count_if(water.begin(), water.end(), [](int i){return i == 0;});
			watershed<vtype>(water, dims);
			int n_empty1 = std::count_if(water.begin(), water.end(), [](int i){return i == 0;});

			sdf(water, voxf.sdf, dims, args.trunc);

			if (args.verbose)  {
				std::cout << "n-empty-before: " << n_empty0 << std::endl;
				std::cout << "n-empty-after: " << n_empty1 << std::endl;
			}

			voxf.pdf.resize(n_elems);
			for (int i = 0; i < n_elems; i++) {
				voxf.pdf[i] = voxf.sdf[i]/args.trunc + 0.5;
				voxf.sdf[i] = voxf.sdf[i]*args.res;
			}

			voxf.save(args.filename_out);
		}




	private:
		Mesh mesh;
};	

void parse_args(int argc, char** argv) {


	cxxopts::Options options("watertight", "Generating sdf grids via watershed.");

	options.add_options()
		("i,in", "input mesh file", cxxopts::value(args.filename_in))
		("o,out", "output file", cxxopts::value(args.filename_out))
		("r,res", "resolution", cxxopts::value(args.res)->default_value("0.002"))
		("t,trunc", "surface truncation distance in voxels", cxxopts::value(args.trunc)->default_value("5"))
		("p,pad", "outer padding for voxelgrid", cxxopts::value(args.pad)->default_value("5"))
		("n,normalize", "normalize to unit cube?", cxxopts::value(args.normalize)->default_value("false"))
		("v,verbose", "verbose", cxxopts::value(args.verbose)->default_value("false"))
		("h,help", "Print usage");


	auto result = options.parse(argc, argv);
	if (result.count("help")) { std::cout << options.help() << std::endl; exit(0); }

	for (auto opt : {"in", "out"}) {
		if (!result.count(opt)) {
			std::cout << options.help() << std::endl;
			cxxopts::throw_or_mimic<cxxopts::option_required_exception>(opt);
			exit(1);
		}
	}

};

int main(int argc, char** argv) {
	parse_args(argc, argv);

	World world;
	world.run();
}
