# pylint: disable=no-member
import onnx
import argparse
from onnx import helper, ValueInfoProto


def set_all_nodes_as_output(input_file, output_file):
    model = onnx.load(input_file)
    orig_outs = [x.name for x in model.graph.output]
    for node in model.graph.node:
        for output in node.output:
            if output not in orig_outs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    onnx.save(model, output_file)
    print(f"{output_file} saved") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', type=str, help='path of input onnx file to process')
    parser.add_argument('output', type=str, help='path of result onnx file')
    subparsers = parser.add_subparsers(help='functions', dest='command')
    extract_parser = subparsers.add_parser('extract', help='extract a subgraph from the onnx file with given'
                                                           'input and output nodes')
    extract_parser.add_argument(
        '-i', '--inodes', nargs='+', help='list of input nodes', required=True)
    extract_parser.add_argument(
        '-o', '--onodes', nargs='+', help='list of output nodes', required=True)
    output_all_parser = subparsers.add_parser(
        'output-all', help='set all nodes as model output(to help debug)')
    args = parser.parse_args()
    command = args.command
    if command == 'extract':
        onnx.utils.extract_model(
            args.file, args.output, args.inodes, args.onodes, check_model=False)
    elif command == 'output-all':
        set_all_nodes_as_output(args.file, args.output)
