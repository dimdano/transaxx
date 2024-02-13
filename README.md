
<div align="center">
  <img width="60%" height="30%" src="./docs/transaxx_logo.png">
</div>

<h1 align="center"> </a></h1>
<h3 align="center">  Fast Emulation of Approximate ViT models in PyTorch  </a></h3>

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Dependencies">Dependencies</a> •
  <a href="#Quick-start">Quick start</a> •
  <a href="#References">References</a> •
  <a href="#Contact">Contact</a>
</p>



## Overview

Current state-of-the-art employs approximate multipliers to address the highly increased power demands of DNN accelerators. However, evaluating the accuracy of approximate DNNs is cumbersome due to the lack of adequate support for approximate arithmetic in DNN frameworks. 

**TransAxx** is a fast emulation framework that extends PyTorch to support approximate inference as well as approximation-aware retraining with GPU acceleration. TransAxx can be seamlessly deployed and is compatible with many DNNs but was built with the aim to emulate Vision Transformer models (ViTs). 


## Dependencies 

* A linux system with docker installed
* An Nvidia GPU card installed that supports the TransAxx docker image (pytorch:2.0.1 and cuda 11.7)
* [Optional]: You can build your own image to fix any dependencies using the provided [build_docker.sh](docker/build_docker.sh) script.
      
## Quick start 

The project has everything installed inside the provided docker image. Run the following commands to get started:

* Run docker container
  
```bash
./docker/run_docker.sh
``` 

* Run jupyter notebook inside docker container (uses port 8888)
```bash
./examples/run_jupyter.sh
``` 
Then copy the jupyter link into a browser and head over the examples folder to run the notebooks

### Using custom multipliers 
* You must use the provided tool [LUT_convert.ipynb](tools/LUT_convert.ipynb) to create a C header file of your custom approximate multiplier.  <br />
 Then you just place it inside the ```ext_modules/include/nn/cuda/axx_mults``` folder. Run the example notebooks to load and evaluate it. <br /> <br />
 **important**: only 8-bit signed multipliers are supported at the moment. So the C header files must include a 256x256 array.

## References

Some libraries of the repo are based on [AdaPT](https://github.com/dimdano/adapt) framework hence the related 'AdaPT' prefixes in the names.

#### Publication

If you use any part of this work, we would love to hear about it and would very much appreciate a citation:

```
@misc{danopoulos2024transaxx,
      title={TransAxx: Efficient Transformers with Approximate Computing}, 
      author={Dimitrios Danopoulos and Georgios Zervakis and Dimitrios Soudris and Jörg Henkel},
      year={2024},
      eprint={2402.07545},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Acknowledgements

This work has been partially funded from the European Union's Horizon Europe research and innovation funding programme under grant agreement No 101070374 CONVOLVE (https://convolve.eu) and by the German Research Foundation (DFG) through the project ``ACCROSS: Approximate Computing aCROss the System Stack'' HE 2343/16-1.

#### Contact 

* Contributor <br/>
`Dimitrios Danopoulos`: dimdano@microlab.ntua.gr


