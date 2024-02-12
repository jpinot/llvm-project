// RUN: %libomp-tdg-compile-and-run
#include <stdio.h>

/* assess the hash table can be extended */

void foo(int i) {
  //printf("i=%d\n", i);
}

int test_many_firstprivates() {
  int var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10\
,var_11,var_12,var_13,var_14,var_15,var_16,var_17,var_18,var_19,var_20\
,var_21,var_22,var_23,var_24,var_25,var_26,var_27,var_28,var_29,var_30\
,var_31,var_32,var_33,var_34,var_35,var_36,var_37,var_38,var_39,var_40\
,var_41,var_42,var_43,var_44,var_45,var_46,var_47,var_48,var_49,var_50\
,var_51,var_52,var_53,var_54,var_55,var_56,var_57,var_58,var_59,var_60\
,var_61,var_62,var_63,var_64,var_65,var_66,var_67,var_68,var_69,var_70\
,var_71,var_72,var_73,var_74,var_75,var_76,var_77,var_78,var_79,var_80\
,var_81,var_82,var_83,var_84,var_85,var_86,var_87,var_88,var_89,var_90\
,var_91,var_92,var_93,var_94,var_95,var_96,var_97,var_98,var_99,var_100\
,var_101,var_102,var_103,var_104,var_105,var_106,var_107,var_108,var_109,var_110\
,var_111,var_112,var_113,var_114,var_115,var_116,var_117,var_118,var_119,var_120\
,var_121,var_122,var_123,var_124,var_125,var_126,var_127,var_128,var_129,var_130\
,var_131,var_132,var_133,var_134,var_135,var_136,var_137,var_138,var_139,var_140\
,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,var_150\
,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160\
,var_161,var_162,var_163,var_164,var_165,var_166,var_167,var_168,var_169,var_170\
,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,var_179,var_180\
,var_181,var_182,var_183,var_184,var_185,var_186,var_187,var_188,var_189,var_190\
,var_191,var_192,var_193,var_194,var_195,var_196,var_197,var_198,var_199,var_200\
,var_201,var_202,var_203,var_204,var_205,var_206,var_207,var_208,var_209,var_210\
,var_211,var_212,var_213,var_214,var_215,var_216,var_217,var_218,var_219,var_220\
,var_221,var_222,var_223,var_224,var_225,var_226,var_227,var_228,var_229,var_230\
,var_231,var_232,var_233,var_234,var_235,var_236,var_237,var_238,var_239,var_240\
,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250\
,var_251,var_252,var_253,var_254,var_255,var_256,var_257,var_258,var_259,var_260\
,var_261,var_262,var_263,var_264,var_265,var_266,var_267,var_268,var_269,var_270\
,var_271,var_272,var_273,var_274,var_275,var_276,var_277,var_278,var_279,var_280\
,var_281,var_282,var_283,var_284,var_285,var_286,var_287,var_288,var_289,var_290\
,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300;
  #pragma omp parallel
  #pragma omp single
  {
    for (int i = 0; i < 10; ++i) {
      #ifdef TDG
      #pragma omp taskgraph recapture(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10\
,var_11,var_12,var_13,var_14,var_15,var_16,var_17,var_18,var_19,var_20\
,var_21,var_22,var_23,var_24,var_25,var_26,var_27,var_28,var_29,var_30\
,var_31,var_32,var_33,var_34,var_35,var_36,var_37,var_38,var_39,var_40\
,var_41,var_42,var_43,var_44,var_45,var_46,var_47,var_48,var_49,var_50\
,var_51,var_52,var_53,var_54,var_55,var_56,var_57,var_58,var_59,var_60\
,var_61,var_62,var_63,var_64,var_65,var_66,var_67,var_68,var_69,var_70\
,var_71,var_72,var_73,var_74,var_75,var_76,var_77,var_78,var_79,var_80\
,var_81,var_82,var_83,var_84,var_85,var_86,var_87,var_88,var_89,var_90\
,var_91,var_92,var_93,var_94,var_95,var_96,var_97,var_98,var_99,var_100\
,var_101,var_102,var_103,var_104,var_105,var_106,var_107,var_108,var_109,var_110\
,var_111,var_112,var_113,var_114,var_115,var_116,var_117,var_118,var_119,var_120\
,var_121,var_122,var_123,var_124,var_125,var_126,var_127,var_128,var_129,var_130\
,var_131,var_132,var_133,var_134,var_135,var_136,var_137,var_138,var_139,var_140\
,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,var_150\
,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160\
,var_161,var_162,var_163,var_164,var_165,var_166,var_167,var_168,var_169,var_170\
,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,var_179,var_180\
,var_181,var_182,var_183,var_184,var_185,var_186,var_187,var_188,var_189,var_190\
,var_191,var_192,var_193,var_194,var_195,var_196,var_197,var_198,var_199,var_200\
,var_201,var_202,var_203,var_204,var_205,var_206,var_207,var_208,var_209,var_210\
,var_211,var_212,var_213,var_214,var_215,var_216,var_217,var_218,var_219,var_220\
,var_221,var_222,var_223,var_224,var_225,var_226,var_227,var_228,var_229,var_230\
,var_231,var_232,var_233,var_234,var_235,var_236,var_237,var_238,var_239,var_240\
,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250\
,var_251,var_252,var_253,var_254,var_255,var_256,var_257,var_258,var_259,var_260\
,var_261,var_262,var_263,var_264,var_265,var_266,var_267,var_268,var_269,var_270\
,var_271,var_272,var_273,var_274,var_275,var_276,var_277,var_278,var_279,var_280\
,var_281,var_282,var_283,var_284,var_285,var_286,var_287,var_288,var_289,var_290\
,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300)
      #endif
      {
        #pragma omp task firstprivate(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10\
,var_11,var_12,var_13,var_14,var_15,var_16,var_17,var_18,var_19,var_20\
,var_21,var_22,var_23,var_24,var_25,var_26,var_27,var_28,var_29,var_30\
,var_31,var_32,var_33,var_34,var_35,var_36,var_37,var_38,var_39,var_40\
,var_41,var_42,var_43,var_44,var_45,var_46,var_47,var_48,var_49,var_50\
,var_51,var_52,var_53,var_54,var_55,var_56,var_57,var_58,var_59,var_60\
,var_61,var_62,var_63,var_64,var_65,var_66,var_67,var_68,var_69,var_70\
,var_71,var_72,var_73,var_74,var_75,var_76,var_77,var_78,var_79,var_80\
,var_81,var_82,var_83,var_84,var_85,var_86,var_87,var_88,var_89,var_90\
,var_91,var_92,var_93,var_94,var_95,var_96,var_97,var_98,var_99,var_100\
,var_101,var_102,var_103,var_104,var_105,var_106,var_107,var_108,var_109,var_110\
,var_111,var_112,var_113,var_114,var_115,var_116,var_117,var_118,var_119,var_120\
,var_121,var_122,var_123,var_124,var_125,var_126,var_127,var_128,var_129,var_130\
,var_131,var_132,var_133,var_134,var_135,var_136,var_137,var_138,var_139,var_140\
,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,var_150\
,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160\
,var_161,var_162,var_163,var_164,var_165,var_166,var_167,var_168,var_169,var_170\
,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,var_179,var_180\
,var_181,var_182,var_183,var_184,var_185,var_186,var_187,var_188,var_189,var_190\
,var_191,var_192,var_193,var_194,var_195,var_196,var_197,var_198,var_199,var_200\
,var_201,var_202,var_203,var_204,var_205,var_206,var_207,var_208,var_209,var_210\
,var_211,var_212,var_213,var_214,var_215,var_216,var_217,var_218,var_219,var_220\
,var_221,var_222,var_223,var_224,var_225,var_226,var_227,var_228,var_229,var_230\
,var_231,var_232,var_233,var_234,var_235,var_236,var_237,var_238,var_239,var_240\
,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250\
,var_251,var_252,var_253,var_254,var_255,var_256,var_257,var_258,var_259,var_260\
,var_261,var_262,var_263,var_264,var_265,var_266,var_267,var_268,var_269,var_270\
,var_271,var_272,var_273,var_274,var_275,var_276,var_277,var_278,var_279,var_280\
,var_281,var_282,var_283,var_284,var_285,var_286,var_287,var_288,var_289,var_290\
,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300)
        {
         foo(var_1);
        }
        #pragma omp task firstprivate(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10\
,var_11,var_12,var_13,var_14,var_15,var_16,var_17,var_18,var_19,var_20\
,var_21,var_22,var_23,var_24,var_25,var_26,var_27,var_28,var_29,var_30\
,var_31,var_32,var_33,var_34,var_35,var_36,var_37,var_38,var_39,var_40\
,var_41,var_42,var_43,var_44,var_45,var_46,var_47,var_48,var_49,var_50\
,var_51,var_52,var_53,var_54,var_55,var_56,var_57,var_58,var_59,var_60\
,var_61,var_62,var_63,var_64,var_65,var_66,var_67,var_68,var_69,var_70\
,var_71,var_72,var_73,var_74,var_75,var_76,var_77,var_78,var_79,var_80\
,var_81,var_82,var_83,var_84,var_85,var_86,var_87,var_88,var_89,var_90\
,var_91,var_92,var_93,var_94,var_95,var_96,var_97,var_98,var_99,var_100\
,var_101,var_102,var_103,var_104,var_105,var_106,var_107,var_108,var_109,var_110\
,var_111,var_112,var_113,var_114,var_115,var_116,var_117,var_118,var_119,var_120\
,var_121,var_122,var_123,var_124,var_125,var_126,var_127,var_128,var_129,var_130\
,var_131,var_132,var_133,var_134,var_135,var_136,var_137,var_138,var_139,var_140\
,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,var_150\
,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160\
,var_161,var_162,var_163,var_164,var_165,var_166,var_167,var_168,var_169,var_170\
,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,var_179,var_180\
,var_181,var_182,var_183,var_184,var_185,var_186,var_187,var_188,var_189,var_190\
,var_191,var_192,var_193,var_194,var_195,var_196,var_197,var_198,var_199,var_200\
,var_201,var_202,var_203,var_204,var_205,var_206,var_207,var_208,var_209,var_210\
,var_211,var_212,var_213,var_214,var_215,var_216,var_217,var_218,var_219,var_220\
,var_221,var_222,var_223,var_224,var_225,var_226,var_227,var_228,var_229,var_230\
,var_231,var_232,var_233,var_234,var_235,var_236,var_237,var_238,var_239,var_240\
,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250\
,var_251,var_252,var_253,var_254,var_255,var_256,var_257,var_258,var_259,var_260\
,var_261,var_262,var_263,var_264,var_265,var_266,var_267,var_268,var_269,var_270\
,var_271,var_272,var_273,var_274,var_275,var_276,var_277,var_278,var_279,var_280\
,var_281,var_282,var_283,var_284,var_285,var_286,var_287,var_288,var_289,var_290\
,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300)
        {
         foo(var_1);
        }
      }
    }
  }
  return 0;
}

int main() {
  return test_many_firstprivates();
}

