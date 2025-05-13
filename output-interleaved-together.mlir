builtin.module {
  aie.device(npu1_1col) {
    %tile-0-1 = aie.tile(0, 1)
    %tile-0-0 = aie.tile(0, 0)
    %tile-0-3 = aie.tile(0, 3)
    %tile-0-2 = aie.tile(0, 2)
    aiex.runtime_sequence(%0 : memref<64x64xi16>, %1 : memref<64x64xi16>, %2 : memref<64x64xi16>) {
      // first part of w to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_W}
      aiex.npu.dma_memcpy_nd(%1[0, 0, 0, 0][1, 2, 32, 32][0, 32, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @of_00_W} : memref<64x64xi16>

      // first input tile to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_I}
      aiex.npu.dma_memcpy_nd(%0[0, 0, 0, 0][1, 2, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @of_00_I} : memref<64x64xi16>

      // second tile of w to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_W}
      aiex.npu.dma_memcpy_nd(%1[0, 0, 0, 2048][1, 2, 32, 32][0, 32, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @of_00_W} : memref<64x64xi16>

      // second input tile to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_I}
      aiex.npu.dma_memcpy_nd(%0[0, 0, 0, 32][1, 2, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @of_00_I} : memref<64x64xi16>

      // first outputs are now ready:
      aiex.npu.dma_memcpy_nd(%2[0, 0, 0, 0][1, 1, 32, 32][0, 0, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @of_02to00_O} : memref<64x64xi16>
      aiex.npu.dma_memcpy_nd(%2[0, 0, 0, 32][1, 1, 32, 32][0, 0, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @of_03to00_O} : memref<64x64xi16>

      // first input tile to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_I}
      aiex.npu.dma_memcpy_nd(%0[0, 0, 0, 2048][1, 2, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @of_00_I} : memref<64x64xi16>

      // second input tile to 02 and 03:
      aiex.npu.dma_wait {symbol = @of_00_I}
      aiex.npu.dma_memcpy_nd(%0[0, 0, 0, 2080][1, 2, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @of_00_I} : memref<64x64xi16>

      // second outputs are now also ready:
      aiex.npu.dma_wait {symbol = @of_02to00_O}
      aiex.npu.dma_memcpy_nd(%2[0, 0, 0, 2048][1, 1, 32, 32][0, 0, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @of_02to00_O} : memref<64x64xi16>
      aiex.npu.dma_wait {symbol = @of_03to00_O}
      aiex.npu.dma_memcpy_nd(%2[0, 0, 0, 2080][1, 1, 32, 32][0, 0, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @of_03to00_O} : memref<64x64xi16>

      // await all fifos
      aiex.npu.dma_wait {symbol = @of_00_W}
      aiex.npu.dma_wait {symbol = @of_00_I}
      aiex.npu.dma_wait {symbol = @of_03to00_O}
      aiex.npu.dma_wait {symbol = @of_02to00_O}
    }
    %3 = aie.core(%tile-0-2) {
      %4 = aie.objectfifo.acquire @of_00to02_W_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %6 = aie.objectfifo.acquire @of_00to02_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %8 = aie.objectfifo.acquire @of_02to00_O_mem(Produce, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @zero_i16(%9) : (memref<32x32xi16>) -> ()
      func.call @matmul_i16_i16(%7, %5, %9) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to02_I_mem(Consume, 1)
      %10 = aie.objectfifo.acquire @of_00to02_W_mem(Consume, 2) : !aie.objectfifosubview<memref<32x32xi16>>
      %11 = aie.objectfifo.subview.access %10[1] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %12 = aie.objectfifo.acquire @of_00to02_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @matmul_i16_i16(%13, %11, %9) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to02_I_mem(Consume, 1)
      aie.objectfifo.release @of_02to00_O_mem(Produce, 1)
      %14 = aie.objectfifo.acquire @of_00to02_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %16 = aie.objectfifo.acquire @of_02to00_O_mem(Produce, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @zero_i16(%17) : (memref<32x32xi16>) -> ()
      func.call @matmul_i16_i16(%15, %5, %17) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to02_I_mem(Consume, 1)
      aie.objectfifo.release @of_00to02_W_mem(Consume, 1)
      %18 = aie.objectfifo.acquire @of_00to02_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @matmul_i16_i16(%19, %11, %17) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to02_I_mem(Consume, 1)
      aie.objectfifo.release @of_02to00_O_mem(Produce, 1)
      aie.objectfifo.release @of_00to02_W_mem(Consume, 1)
      aie.end
    } { link_with="mm_32x32x32.o" }
    %20 = aie.core(%tile-0-3) {
      %21 = aie.objectfifo.acquire @of_00to03_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %23 = aie.objectfifo.acquire @of_00to03_W_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %25 = aie.objectfifo.acquire @of_03to00_O_mem(Produce, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @zero_i16(%26) : (memref<32x32xi16>) -> ()
      func.call @matmul_i16_i16(%22, %24, %26) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to03_I_mem(Consume, 1)
      %27 = aie.objectfifo.acquire @of_00to03_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %29 = aie.objectfifo.acquire @of_00to03_W_mem(Consume, 2) : !aie.objectfifosubview<memref<32x32xi16>>
      %30 = aie.objectfifo.subview.access %29[1] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @matmul_i16_i16(%28, %30, %26) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to03_I_mem(Consume, 1)
      aie.objectfifo.release @of_03to00_O_mem(Produce, 1)
      %31 = aie.objectfifo.acquire @of_00to03_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      %33 = aie.objectfifo.acquire @of_03to00_O_mem(Produce, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @zero_i16(%34) : (memref<32x32xi16>) -> ()
      func.call @matmul_i16_i16(%32, %24, %34) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to03_I_mem(Consume, 1)
      aie.objectfifo.release @of_00to03_W_mem(Consume, 1)
      %35 = aie.objectfifo.acquire @of_00to03_I_mem(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
      %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
      func.call @matmul_i16_i16(%36, %30, %34) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
      aie.objectfifo.release @of_00to03_I_mem(Consume, 1)
      aie.objectfifo.release @of_03to00_O_mem(Produce, 1)
      aie.objectfifo.release @of_00to03_W_mem(Consume, 1)
      aie.end
    } { link_with="mm_32x32x32.o" }

    // weight fifos: (combined with distribute pattern)
    aie.objectfifo @of_00_W(%tile-0-0, {%tile-0-1}, 2 : i32) : !aie.objectfifo<memref<2x32x32xi16>>
    aie.objectfifo @of_00to02_W_mem(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile-0-2}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @of_00to03_W_mem(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile-0-3}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@of_00_W] -> [@of_00to02_W_mem, @of_00to03_W_mem]([] [0, 1024])

    // input fifos: (combined with distribute pattern)
    aie.objectfifo @of_00_I(%tile-0-0, {%tile-0-1}, 1 : i32) : !aie.objectfifo<memref<2x32x32xi16>>
    aie.objectfifo @of_00to02_I_mem(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile-0-2}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @of_00to03_I_mem(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile-0-3}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@of_00_I] -> [@of_00to02_W_mem, @of_00to03_I_mem]([] [0, 1024])

    // output fifos: (split per tile)
    aie.objectfifo @of_02to00_O_mem(%tile-0-2, {%tile-0-1}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @of_02to00_O(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%tile-0-0}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@of_02to00_O_mem] -> [@of_02to00_O]([] [])

    aie.objectfifo @of_03to00_O_mem(%tile-0-3, {%tile-0-1}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @of_03to00_O(%tile-0-1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%tile-0-0}, 1 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@of_03to00_O_mem] -> [@of_03to00_O]([] [])

    // function declarations:
    func.func private @matmul_i16_i16(memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    func.func private @zero_i16(memref<32x32xi16>) -> ()
  }
}
