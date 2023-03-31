.PHONY: build

build:
	if ! [ -d "build" ]; then \
		mkdir build; \
	fi
	
	cd build && cmake .. \
		-DCMAKE_C_COMPILER=/usr/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/bin/g++ \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE="/mnt/c/Users/ChaCha/Desktop/git/linux_vcpkg/scripts/buildsystems/vcpkg.cmake"
		
	cd build && make -j$(shell nproc)
		
